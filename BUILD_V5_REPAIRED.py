from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
STAGE = ROOT / "build" / ("v5_stage_" + STAMP)
BUNDLE = STAGE / "dist" / "TyreTrack"
RELEASE = ROOT / "releases" / ("TyreTrack_V5_" + STAMP)

def stop(message):
    raise SystemExit("STOP: " + message)

def run(command):
    print("\nRUN:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def main():
    if sys.platform != "win32":stop("Run on Windows 11.")
    if sys.version_info[:2] != (3,12):stop("Run with Python 3.12 from .buildvenv.")
    required=["TyreTrack_Production_Monitor.py","count_line.py","hourly_dashboard.py","tyre_detector.pt"]
    for name in required:
        if not (ROOT/name).is_file():stop("Missing "+name)
    run([sys.executable,"-m","py_compile",*[ROOT/name for name in required if name.endswith(".py")]])
    import torch, torchvision, depthai, ultralytics, PyQt5
    from ultralytics import YOLO
    print("torch:",torch.__version__,"torchvision:",torchvision.__version__,"depthai:",depthai.__version__)
    print("Torch test:",torch.zeros(1))
    # V5 code may require the DepthAI 2.x API; refuse an incompatible SDK.
    source=(ROOT/"TyreTrack_Production_Monitor.py").read_text(encoding="utf-8-sig")
    if "dai.node.XLinkOut" in source and not hasattr(depthai.node,"XLinkOut"):
        stop("Source needs DepthAI 2.x, but installed SDK lacks XLinkOut.")
    model=YOLO(str(ROOT/"tyre_detector.pt"))
    if model.task!="detect":stop("Model is not a detection model.")
    del model
    STAGE.mkdir(parents=True,exist_ok=False)
    command=[
        sys.executable,"-m","PyInstaller","--noconfirm","--clean",
        "--onedir","--console","--name","TyreTrack",
        "--distpath",STAGE/"dist","--workpath",STAGE/"work",
        "--specpath",STAGE/"spec",
        "--paths",ROOT,
        "--hidden-import","count_line",
        "--hidden-import","hourly_dashboard",
        "--hidden-import","ultralytics.nn.tasks",
        "--hidden-import","ultralytics.trackers.byte_tracker",
        "--collect-data","ultralytics",
        "--collect-binaries","torchvision",
        "--copy-metadata","torchvision",
        "--collect-all","depthai",
        "--exclude-module","tensorflow",
        "--exclude-module","keras",
        "--exclude-module","tf_keras",
        "--exclude-module","onnx",
        "--exclude-module","onnxruntime",
        "--exclude-module","onnxslim",
        "--add-data",str(ROOT/"tyre_detector.pt")+";.",
        ROOT/"TyreTrack_Production_Monitor.py"
    ]
    run(command)
    exe=BUNDLE/"TyreTrack.exe"
    internal=BUNDLE/"_internal"
    if not exe.is_file() or not internal.is_dir():stop("Incomplete PyInstaller bundle.")
    # Match previous known-good fix: only remove Qt-local duplicates if
    # the root copy is present. NEVER remove PyTorch's torch/lib DLLs.
    for qtbin in (internal/"PyQt5"/"Qt5"/"bin",internal/"PyQt5"/"Qt"/"bin"):
        for name in ("MSVCP140.dll","VCRUNTIME140.dll","VCRUNTIME140_1.dll"):
            local=qtbin/name
            root=internal/name
            if local.is_file() and root.is_file():
                backup=STAGE/"qt_dll_backup"/qtbin.relative_to(internal)/name
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(local),str(backup))
                print("Moved duplicate Qt DLL to backup:",local)
    # Previous successful builder copied a specifically named optree DLL.
    # Use detected wheel filename rather than hardcoding its hash.
    try:
        import optree
        optree_dir=Path(optree.__file__).resolve().parent
        libs=optree_dir.parent/"optree.libs"
        if libs.is_dir():
            for source_dll in libs.glob("msvcp140-*.dll"):
                for target_dir in (internal/"optree",internal/"optree.libs"):
                    target_dir.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source_dll,target_dir/source_dll.name)
                    if digest(source_dll)!=digest(target_dir/source_dll.name):
                        stop("optree DLL copy checksum mismatch.")
                    print("Verified optree DLL:",target_dir/source_dll.name)
    except ImportError:
        print("optree not installed as direct import; skipping optional optree workaround.")
    weights=list(BUNDLE.rglob("tyre_detector.pt"))
    if len(weights)!=1 or digest(weights[0])!=digest(ROOT/"tyre_detector.pt"):
        stop("Detector model missing, duplicated or modified.")
    if list(BUNDLE.rglob("email_config.ini")):
        stop("Private email_config.ini included in bundle.")
    # No copy of old SQLite database: V5 is intended to start with zero counts.
    (BUNDLE/"data").mkdir(exist_ok=True)
    (BUNDLE/"reports").mkdir(exist_ok=True)
    print("\nSTARTUP SMOKE TEST: launch and watch 10 seconds...")
    with (STAGE/"startup_output.txt").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([str(exe)],cwd=BUNDLE,stdout=log,stderr=subprocess.STDOUT)
        try:
            time.sleep(10)
            code=proc.poll()
            if code is not None:
                stop("EXE exited during startup with code "+str(code)+". See "+str(STAGE/"startup_output.txt"))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    RELEASE.parent.mkdir(exist_ok=True)
    os.rename(BUNDLE,RELEASE)
    print("\nSTARTUP TEST PASSED; NEW RELEASE:",RELEASE)
    print("Test a recorded video before shipping. Live OAK-D and SMTP remain untested.")
    print("Working old application has not been touched.")

if __name__=="__main__":
    main()
