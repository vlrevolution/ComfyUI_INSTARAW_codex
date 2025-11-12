from server import PromptServer
from aiohttp import web
from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted
import time, json, os, hashlib
from typing import Optional, List, Dict

REQUEST_RESHOW = "-1"
CANCEL = "-3"
WAITING_FOR_RESPONSE = "-9"

SPECIALS = [REQUEST_RESHOW, CANCEL, WAITING_FOR_RESPONSE]

class Response:
    def __init__(self, selection:Optional[List[str]] = None, text:Optional[str] = None,
                        masked_image:Optional[str] = None, extras:Optional[List[str]] = None,
                        crop:Optional[Dict] = None): # Add the new crop parameter
        self.selection:List[int]        = [int(x) for x in selection] if selection else []
        self.text:Optional[str]         = text
        self.masked_image:Optional[str] = masked_image
        self.extras:Optional[List[str]] = extras
        self.crop:Optional[Dict]        = crop # Store the crop data

    def get_extras(self,defaults:list[str]) -> list[str]:
         return self.extras or defaults  

class TimeoutResponse(Response): pass
class CancelledResponse(Response): pass
class RequestResponse(Response): pass

class MessageState:
    _latest:'Optional[MessageState]' = None
    unique_expected = None

    def __init__(self, data:dict|str={}):
        data_dict:dict = data if isinstance(data,dict) else json.loads(data)
        self.unique:str            = data_dict.pop('unique', None)
        self.special:Optional[str] = data_dict.pop('special',None)
        self.response:Response     = Response(**data_dict)

    @classmethod
    def latest(cls) -> 'MessageState': 
        if cls._latest is None: cls._latest = cls()
        return cls._latest 
    
    @classmethod
    def set_latest(cls, latest:'MessageState'):
        cls._latest = latest

    @classmethod
    def waiting_state(cls): return MessageState(data={'special':WAITING_FOR_RESPONSE})

    @classmethod
    def request_state(cls): return MessageState(data={'special':REQUEST_RESHOW})

    @classmethod
    def start_waiting(cls, unique): 
        cls._latest = cls.waiting_state()
        cls.unique_expected = unique

    @classmethod
    def get_response(cls) -> Response:  
        if cls.waiting(): return TimeoutResponse()
        if cls.latest().cancelled: return CancelledResponse()
        if cls.latest().request: return RequestResponse()
        return cls.latest().response

    @classmethod
    def stop_waiting(cls): 
        cls._latest = MessageState()

    @classmethod
    def waiting(cls) -> bool: return cls.latest().special == WAITING_FOR_RESPONSE

    @property
    def cancelled(self) -> bool: return self.special == CANCEL

    @property
    def request(self) -> bool: return self.special == REQUEST_RESHOW

    @property
    def real(self) -> bool: return self.special is None


@PromptServer.instance.routes.post('/instaraw/interactive_message')
async def cg_image_filter_message(request):
    post     = await request.post()
    response = post.get("response")
    message  = MessageState(response)

    if str(MessageState.unique_expected)==str(message.unique):
        if (MessageState.waiting()):
            MessageState.set_latest(message)
        else:
            print(f"Ignoring response {response} because not waiting for one")
    else:
        print(f"Ignoring mismatched response {response}")

    return web.json_response({})

# --- NEW API ENDPOINT FOR INSTANT CACHE CLEARING ---
@PromptServer.instance.routes.post('/instaraw/clear_text_filter_cache')
async def clear_text_filter_cache(request):
    try:
        # The UID from the request isn't strictly needed anymore, but we can keep it for logging.
        data = await request.json()
        uid = data.get('uid')
        print(f"🧹 INSTARAW API: Received request to clear text filter cache from node UID {uid}.")

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        if not os.path.isdir(cache_dir):
            print("🧹 INSTARAW API: Cache directory does not exist. Nothing to clear.")
            return web.json_response({"status": "success", "cleared_count": 0, "message": "Cache directory not found."})

        cleared_count = 0
        files_cleared = []
        
        # Iterate over all files in the cache directory
        for filename in os.listdir(cache_dir):
            # If a file is a text filter cache file, delete it
            if filename.endswith("_text_edit.json"):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    files_cleared.append(filename)
                    cleared_count += 1
                except OSError as e:
                    print(f"⚠️ INSTARAW API: Error clearing cache file {file_path}: {e}")
                    # We continue even if one file fails to delete
        
        print(f"✅ INSTARAW API: Successfully cleared {cleared_count} text filter cache file(s).")
        return web.json_response({"status": "success", "cleared_count": cleared_count, "files": files_cleared})

    except Exception as e:
        print(f"❌ INSTARAW API: An unexpected error occurred while clearing cache: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


def wait_for_response(secs, uid, unique) -> Response:
    MessageState.start_waiting(unique)
    try:
        end_time = time.monotonic() + secs
        while(time. monotonic() < end_time and MessageState.waiting()): 
            throw_exception_if_processing_interrupted()
            PromptServer.instance.send_sync("instaraw-interactive-images", {"tick": int(end_time - time.monotonic()), "uid": uid, "unique":unique})
            time.sleep(0.5)
        if MessageState.waiting():
            PromptServer.instance.send_sync("instaraw-interactive-images", {"timeout": True, "uid": uid, "unique":unique})
        return MessageState.get_response()
    finally: MessageState.stop_waiting()
    
def send_and_wait(payload, timeout, uid, unique) -> Response:
    payload['uid'] = uid
    payload['unique'] = unique

    while True:
        PromptServer.instance.send_sync("instaraw-interactive-images", payload)
        r = wait_for_response(timeout, uid, unique)
        if isinstance(r,CancelledResponse): raise InterruptProcessingException()
        if (not isinstance(r, RequestResponse)): return r