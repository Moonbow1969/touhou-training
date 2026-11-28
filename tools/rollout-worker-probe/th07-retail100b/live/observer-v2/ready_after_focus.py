from pathlib import Path
p=Path("tool/start.sh")
s=p.read_text()
old="""        echo "$window" > /capture/window
        xdotool windowfocus "$window"
        break"""
new="""        if xdotool windowfocus "$window"; then
            echo "$window" > /capture/window.ready
            mv /capture/window.ready /capture/window
            break
        fi"""
assert s.count(old)==1
tmp=p.with_suffix('.ready')
tmp.write_text(s.replace(old,new));tmp.chmod(p.stat().st_mode);tmp.replace(p)
