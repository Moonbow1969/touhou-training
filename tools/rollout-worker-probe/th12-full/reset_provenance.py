"""Verify the physical Wine identity and the realized path of an explicitly requested reset."""
import hashlib

def inspect_reset(raw, requested):
    def content(name):return (raw/name).read_text().strip()
    def digest(name):return hashlib.sha256((raw/name).read_bytes()).hexdigest()
    modes=dict(line.split('=',1) for line in content('reset-mode.txt').splitlines())
    effective=modes['effective'];before=content('start-process.txt');after=content('new-story-process.txt')
    if modes['requested']!=requested or effective not in ({'warm','cold_fallback'} if requested=='warm' else {'cold'}):raise ValueError('Unregistered original reset path')
    if len(after.split())!=2 or not all(k.isdigit() for k in after.split()) or after!=content('terminal-process.txt'):raise ValueError('The native process changed during the new credit')
    result=dict(requested=requested,effective=effective,pid_and_start_before=before,pid_and_start_in_new_game=after,record_sha256=digest('reset-mode.txt'))
    if effective=='warm':
        lines=content('preceding-terminal.tsv').splitlines()
        if len(lines)!=6 or lines[1]!=before or before!=after:raise ValueError('Missing preceding native terminal or original process continuity')
        result.update(previous_episode=lines[0],preceding_terminal_record_sha256=digest('preceding-terminal.tsv'),rank_cursor_on_warm_entry=int(content('rank-menu-as-entered.txt')))
    if effective=='cold_fallback':
        if before==after or not (raw/'warm-failure.txt').is_file():raise ValueError('Unverified fallback to a fresh game')
        result.update(warm_failure_code=content('warm-failure.txt'),warm_navigation_log_sha256=digest('warm-navigation.log'))
    return result


def verify_predecessor(current_raw, preceding_raw):
    """Re-read the previous original terminal files independently of the collector."""
    lines=(current_raw/'preceding-terminal.tsv').read_text().splitlines()
    if len(lines)!=6 or lines[0]!=preceding_raw.name:raise ValueError('Preceding native episode differs from the recorded lane order')
    before=(current_raw/'start-process.txt').read_text().strip()
    prior=(preceding_raw/'terminal-process.txt').read_text().strip()
    if lines[1]!=before or lines[1]!=prior:raise ValueError('Previous terminal process differs from the current original process')
    files={}
    for record in lines[2:]:
        expected,path=record.split(maxsplit=1);name=path.rsplit('/',1)[-1]
        if name in files or path!=f'/capture/full/{preceding_raw.name}/{name}':raise ValueError('Predecessor trace points to another capture')
        actual=hashlib.sha256((preceding_raw/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Changed preceding original {name}')
        files[name]=actual
    if set(files)!={'outcome.txt','terminal-at-control.ppm','engineering-native.tsv','difficulty-native.tsv'}:raise ValueError('Missing preceding stock, rank, terminal image or outcome')
    return files
