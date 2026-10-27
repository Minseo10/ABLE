"""Select reference regions in a laptop browser. Standard library only, no robot IO."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PAGE = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sprout first-time setup</title>
<style>
body{font:16px system-ui;max-width:900px;margin:30px auto;padding:0 20px;color:#1f2937}
h1{font-size:24px}canvas{width:100%;height:auto;border:1px solid #94a3b8;touch-action:none;cursor:crosshair}
button,select,input{font:inherit;padding:8px;margin:6px 8px 6px 0}button{cursor:pointer}
#message{white-space:pre-wrap;color:#9f1239}#help{min-height:48px}small{display:block;color:#475569}
</style>
<h1>Initialize Sprout tracking</h1>
<p>Draw boxes on this new camera capture. This saves tracking settings only. It does not move the robot.</p>
<p id="help"></p>
<label id="cameraLabel">Forearm camera <select id="camera"><option value="0">Camera 0 (upper view)</option><option value="1">Camera 1 (lower view)</option></select></label>
<canvas id="canvas"></canvas>
<label>Floor to top of forearm at tape: <input id="height" type="number" min="0.1" max="29.9" step="0.1" value="8" style="width:80px"> cm</label>
<small>8 cm is the current experiment assumption. Change it if your setup differs.</small>
<div><button id="back">Back</button><button id="clear">Redraw</button><button id="next">Next</button><button id="cancel">Cancel</button></div>
<p id="message"></p>
<script>
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const camera=document.getElementById('camera'), next=document.getElementById('next');
const message=document.getElementById('message');
const captions=[
'Draw a tight box around the WHITE TAPE ON SPROUT’S LEFT FOREARM, including nearby forearm texture. Do not select the white floor tape.',
'Draw a box enclosing ALL visible Sprout parts in camera 0, including the target forearm. Exclude TidyBot. This supplies a coarse body obstacle.',
'Draw a box enclosing ALL visible Sprout parts in camera 1, including the feet. Exclude TidyBot. Both views are required by this body tracker.'
];
let step=0, rects=[null,null,null], anchor=null, loaded=false;
const img=new Image();
function view(){return step===0?Number(camera.value):step-1;}
function draw(){
 if(!loaded)return;
 ctx.drawImage(img,0,view()*canvas.height,canvas.width,canvas.height,0,0,canvas.width,canvas.height);
 const r=rects[step]; if(r){ctx.strokeStyle=step===0?'#00ff88':'#ff8c00';ctx.lineWidth=3;ctx.strokeRect(...r);}
 document.getElementById('help').textContent=captions[step];
 document.getElementById('cameraLabel').style.display=step===0?'inline':'none';
 document.getElementById('back').disabled=step===0;
 next.textContent=step===2?'Save reference':'Next'; next.disabled=!r;
}
img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight/2;loaded=true;draw();};
img.onerror=()=>{message.textContent='Cannot load the captured camera image.';};
img.src='image';
function point(e){const b=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,Math.round((e.clientX-b.left)*canvas.width/b.width))),Math.max(0,Math.min(canvas.height,Math.round((e.clientY-b.top)*canvas.height/b.height)))];}
canvas.onpointerdown=e=>{if(!loaded)return;anchor=point(e);canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!anchor)return;const p=point(e);rects[step]=[Math.min(p[0],anchor[0]),Math.min(p[1],anchor[1]),Math.abs(p[0]-anchor[0]),Math.abs(p[1]-anchor[1])];draw();};
canvas.onpointerup=()=>{anchor=null;const r=rects[step];if(r&&(r[2]<8||r[3]<8)){rects[step]=null;message.textContent='Draw a box at least 8 pixels wide and high.';}draw();};
camera.onchange=()=>{rects[0]=null;draw();};
document.getElementById('clear').onclick=()=>{rects[step]=null;draw();};
document.getElementById('back').onclick=()=>{step=Math.max(0,step-1);message.textContent='';draw();};
async function post(value){const response=await fetch('selection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});if(!response.ok)throw Error(await response.text());return response;}
next.onclick=async()=>{
 message.textContent='';if(step<2){step++;draw();return;}
 const height=Number(document.getElementById('height').value)/100;
 if(!(height>0&&height<0.3)){message.textContent='Enter a height between 0 and 30 cm.';return;}
 next.disabled=true;
 try{await post({camera_index:Number(camera.value),patch_xywh:rects[0],body_envelopes_xywh:rects.slice(1),top_height_m:height});message.textContent='Selections sent. Return to the terminal for validation. You may close this tab.';}
 catch(e){message.textContent=String(e);next.disabled=false;}
};
document.getElementById('cancel').onclick=async()=>{try{await post({cancel:true});message.textContent='Cancelled. You may close this tab.';}catch(e){message.textContent=String(e);}};
</script>"""


def selection_server(image: bytes) -> tuple[HTTPServer, str, dict]:
    """Bind only loopback and accept writes only from this unguessable page."""
    token = secrets.token_urlsafe(24)
    state: dict = {}
    base = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status: int, data: bytes, mime: str = "text/plain"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == base:
                self.respond(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == base + "image":
                self.respond(200, image, "image/jpeg")
            else:
                self.respond(404, b"Not found")

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.path != base + "selection" or self.headers.get("Origin") != origin:
                self.respond(403, b"Invalid selection origin")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < size <= 8192
                    or self.headers.get_content_type() != "application/json"
                ):
                    raise ValueError("Invalid selection request")
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict):
                    raise TypeError("Expected selection object")
            except (ValueError, TypeError) as error:
                self.respond(400, str(error).encode())
                return
            state["selection"] = value
            self.respond(200, b"Saved")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.5
    return server, f"http://127.0.0.1:{server.server_port}{base}", state


def select_reference(image_path: Path, output: Path) -> None:
    data = image_path.read_bytes()
    server, url, state = selection_server(data)
    print(
        f"Select the taped forearm and body regions in your browser: {url}", flush=True
    )
    try:
        webbrowser.open(url)
        deadline = time.monotonic() + 900
        while "selection" not in state:
            if time.monotonic() >= deadline:
                raise ValueError("Reference selection timed out")
            server.handle_request()
        result = state["selection"]
        if result.get("cancel"):
            raise ValueError("Reference selection cancelled")
        result["image_sha256"] = hashlib.sha256(data).hexdigest()
        output.write_text(json.dumps(result) + "\n")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        select_reference(args.image, args.output)
    except ValueError as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
