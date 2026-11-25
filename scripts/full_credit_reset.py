"""Start and verify fresh Normal Reimu A credits in stock TH14."""

from time import monotonic, sleep


def _native_state(game, reader):
    offsets = reader.offsets

    def read(offset):
        return reader.read_uint(offset, relative=True)

    thread = offsets.game_thread
    pointer = read(thread.game_thread_pointer)
    return {
        "screen": read(offsets.statics.game_screen),
        "state": game.read_game_val("game_state"),
        "stage": read(offsets.environment.stage),
        "difficulty": read(offsets.environment.difficulty),
        "character": read(offsets.environment.character),
        "subshot": read(offsets.environment.subshot),
        "chapter": read(offsets.statics.stage_chapter),
        "stage_frame": reader.read_uint(pointer + thread.zGameThread_stage_timer) if pointer else 0,
        "continues": read(offsets.statics.continues),
        "score": game.read_game_val("score"),
        "resources": {key: game.read_game_val(key) for key in
                      ("lives", "bombs", "power", "life_fragments", "bomb_fragments")},
    }


def _require_story(state):
    if (state["screen"], state["difficulty"], state["character"], state["subshot"]) != (7, 1, 0, 0):
        raise RuntimeError(f"Expected stock TH14 Normal Reimu A story gameplay or menu, got {state}.")


def _is_fresh(state):
    return (state["state"] == 2 and state["stage"] == 1 and state["continues"] == 0
            and state["score"] == 0 and state["resources"]["lives"] == 2
            and state["resources"]["bombs"] == 3
            and all(value is not None for value in state["resources"].values()))


def _require_fresh(state):
    _require_story(state)
    if not _is_fresh(state):
        raise RuntimeError(f"Expected a fresh zero-Continue Stage 1 credit, got {state}.")


def _recover_menu(game, reader, before, timeout):
    _require_story(before)
    if before["state"] == 1 and before["stage"] == 1 and (before["resources"]["lives"] or 0) < 0:
        key, route = "z", "game_over_retry"
    elif before["state"] == 0 and before["stage"] == 1:
        key, route = "esc", "resume_pause"
    else:
        raise RuntimeError(f"No verified fresh-credit menu route for {before}.")

    game.require_game_focus()
    game.resume_game_process()
    try:
        game.release_all_keys()
        if route == "game_over_retry":
            # A newly opened Game Over menu starts on the top Retry item.
            # Up wraps to Give Up and Retry, which opens a new zero-Continue credit.
            game._sleep(30, poll_interval=.001)
            game._press_and_release("up")
            game._sleep(8, poll_interval=.001)
        game._press_and_release(key)
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            game.require_game_focus()
            if game.read_game_val("game_state") == 2:
                state = _native_state(game, reader)
                _require_story(state)
                if state["state"] == 2:
                    if route == "game_over_retry":
                        _require_fresh(state)
                    return route, state
            sleep(.01)
        raise TimeoutError(f"TH14 did not enter active gameplay after {route}.")
    finally:
        game.release_all_keys()


def prepare_fresh_credit(parakit, *, timeout=30.):
    """Recover an initial Game Over before constructing the active-game adapter."""
    from environment import interface as game
    from environment.structured import StateReader, load_parakit_offsets

    reader = StateReader(game, load_parakit_offsets(parakit), 96)
    game._get_focus()
    before = _native_state(game, reader)
    _require_story(before)
    if before["state"] == 2:
        return {"route": "already_active", "before": before, "opening": before}
    route, opening = _recover_menu(game, reader, before, timeout)
    return {"route": route, "before": before, "opening": opening}


def _reset_adapter(env, adopt):
    # Reuse the adapter's prepared-first-reset path and keep its lifetime count.
    prepared, count = env.prepared, env.reset_count
    env.prepared = adopt
    if adopt:
        env.reset_count = 0
    try:
        result = env.reset()
    except BaseException:
        env.reset_count = count
        raise
    else:
        if adopt:
            env.reset_count += count
        return result
    finally:
        env.prepared = prepared


def _wait_for_retry_shortcut(env, timeout):
    # Native Story Retry was qualified after the opening reached frame 180.
    if env.stage_frame() >= 180:
        return
    game = env.game
    deadline = monotonic() + timeout
    game.resume_game_process()
    try:
        while env.stage_frame() < 180:
            game.require_game_focus()
            if monotonic() >= deadline:
                raise TimeoutError("TH14 did not reach its qualified active Retry window.")
            sleep(.005)
    finally:
        env.freeze()


def fresh_credit_reset(env, trial=None, *, start_frame=30, timeout=30.):
    """Return the adapter observation and a verified opening trace for one trial."""
    if env.stage_practice or (env.first_stage, env.difficulty) != (1, 1):
        raise ValueError("A fresh credit requires the Normal Stage 1 story adapter.")
    env.wait_for_focus()
    env.release()
    game = env.game
    before = _native_state(game, env.reader)
    _require_story(before)
    route = "native_retry"
    state = before
    if before["state"] == 1 or before["state"] == 0 and before["stage"] == 1:
        route, state = _recover_menu(game, env.reader, before, timeout)
        env.freeze()
    adopt = ((route == "game_over_retry" or not env.reset_count) and _is_fresh(state)
             and (start_frame is None or state["stage_frame"] <= start_frame))
    if adopt and route == "native_retry":
        route = "prepared_opening"
    if not adopt and state["stage"] == 1 and state["state"] == 2:
        _wait_for_retry_shortcut(env, timeout)
    observation, info = _reset_adapter(env, adopt)
    _require_fresh(_native_state(game, env.reader))
    if start_frame is not None and env.stage_frame() != start_frame:
        from scripts.benchmark_native_clock import align_start

        align_start(env, start_frame)
        env.history.clear()
        env.remember()
        observation = env.observation()
    opening = _native_state(game, env.reader)
    _require_fresh(opening)
    return observation, dict(info, fresh_credit_reset={"route": route, "before": before, "opening": opening})
