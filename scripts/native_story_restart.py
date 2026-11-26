"""Visually qualify stock TH14 Game Over Retry before confirming it."""

from time import monotonic, sleep

import numpy as np
from PIL import Image, ImageFilter

from scripts import full_credit_reset


class FreshCreditRequired(RuntimeError):
    """The current menu has no qualified route to an unused stock credit."""

    def __init__(self, state, message):
        super().__init__(message)
        self.state = state


class GameOverMenu:
    def __init__(self, default, retry, crops, maximum_error):
        self.crops = crops
        self.maximum_error = maximum_error
        with Image.open(default) as image:
            self.default = self.pixels(image)
        with Image.open(retry) as image:
            self.retry = self.pixels(image)
        if sum(error > 2 * self.maximum_error for error in self.errors(self.default, self.retry)) < 2:
            raise ValueError("Provide distinct crops covering both the first and last selected rows.")

    def pixels(self, image):
        crops = []
        for x, y, width, height in self.crops:
            if width < 1 or height < 1 or x < 0 or y < 0 or x + width > image.width or y + height > image.height:
                raise ValueError("Each Game Over menu crop must fit the screenshot.")
            gray = image.crop((x, y, x + width, y + height)).convert("L")
            # The glyphs are high frequency while the frozen playfield background is not stable across stages.
            smooth = gray.filter(ImageFilter.GaussianBlur(1.5))
            crops.append((np.asarray(gray, dtype=np.float32) - np.asarray(smooth, dtype=np.float32)) / 255)
        return crops

    @staticmethod
    def errors(a, b):
        return [float(np.abs(left - right).mean()) for left, right in zip(a, b, strict=True)]

    def selection(self, image):
        observed = self.pixels(image)
        errors = {"default": self.errors(self.default, observed), "retry": self.errors(self.retry, observed)}
        matches = [label for label, per_crop in errors.items() if all(error <= self.maximum_error for error in per_crop)]
        return (matches[0] if len(matches) == 1 else None), errors


def game_over(state):
    return (state["screen"] == 7 and state["state"] == 1 and state["stage"] in range(1, 7)
            and state["resources"]["lives"] is not None and state["resources"]["lives"] < 0
            and state["continues"] == 0)


def coherent_selection(game, reader, menu, timeout, required=()):
    deadline = monotonic() + timeout
    previous = None
    errors = {}
    while monotonic() < deadline:
        game.require_game_focus()
        state = full_credit_reset._native_state(game, reader)
        full_credit_reset._require_story(state)
        if not game_over(state):
            raise FreshCreditRequired(state, "The expected zero-Continue Game Over menu changed.")
        label, errors = menu.selection(game.capture_frame())
        if label and label == previous and (not required or label in required):
            return label, errors
        previous = label
        sleep(.08)
    raise FreshCreditRequired(state, f"Game Over menu reference did not match twice: {errors}.")


MENU_ROW_TOPS = (258, 284, 309, 335, 361)

def moved_menu_cursor(before, after):
    """Return (departed, selected) on a clear transition between the five game-over rows."""
    previous = np.asarray(before.convert("RGB"), dtype=np.int16)
    current = np.asarray(after.convert("RGB"), dtype=np.int16)
    if previous.shape != (448, 384, 3) or current.shape != previous.shape:
        raise ValueError("Expected TH14's 384x448 playable-area capture.")
    delta = current[:, :, 0] - previous[:, :, 0]
    brighter, darker = [], []
    for y in MENU_ROW_TOPS:
        text = delta[y:y + 17, 46:86]
        brighter.append(np.count_nonzero(text > 55))
        darker.append(np.count_nonzero(text < -55))

    def unique_maximum(counts):
        high, next_high = sorted(counts)[-2:][::-1]
        return counts.index(high) if high >= 80 and high - next_high >= 55 else None

    departed, selected = unique_maximum(darker), unique_maximum(brighter)
    if departed is None or selected is None or departed == selected:
        return None
    return departed, selected

def find_retry_by_cursor(game, reader):
    for _ in range(5):
        game.require_game_focus()
        if not game_over(full_credit_reset._native_state(game, reader)):
            break
        previous = game.capture_frame()
        game._press_and_release("up")
        game._sleep(10, poll_interval=.001)
        if moved_menu_cursor(previous, game.capture_frame()) == (0, 4):
            if game_over(full_credit_reset._native_state(game, reader)):
                return {"observed_cursor_transition": [0, 4]}
    raise FreshCreditRequired(full_credit_reset._native_state(game, reader), "Could not verify the last menu row.")

def retry_game_over(game, reader, state, menu, timeout=10.):
    full_credit_reset._require_story(state)
    if not game_over(state):
        raise FreshCreditRequired(state, "Only an actual zero-Continue Game Over may use this menu route.")
    game.require_game_focus()
    game.release_all_keys()
    game.resume_game_process()
    try:
        first_match = None
        try:
            label, first_match = coherent_selection(game, reader, menu, min(timeout, 4.))
            if label == "default":
                game._press_and_release("up")
            _, confirmed_match = coherent_selection(game, reader, menu, min(timeout, 4.), required=("retry",))
        except FreshCreditRequired:
            confirmed_match = find_retry_by_cursor(game, reader)
        game.require_game_focus()
        game._press_and_release("z")
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            game.require_game_focus()
            after = full_credit_reset._native_state(game, reader)
            if after["state"] == 2:
                full_credit_reset._require_fresh(after)
                return dict(route="visually_verified_game_over_retry", before=state, opening=after,
                            initial_reference_errors=first_match, retry_reference_errors=confirmed_match)
            sleep(.02)
        raise FreshCreditRequired(state, "The verified Retry did not produce a new stock credit.")
    finally:
        game.release_all_keys()


def prepare(game, reader, menu, restart_hook=None):
    """Get the native runtime to a verified active state without scoring an episode."""
    game._get_focus()
    state = full_credit_reset._native_state(game, reader)
    if state["screen"] == 7 and state["state"] == 2 and not state["continues"]:
        full_credit_reset._require_story(state)
        return None
    if menu is not None and game_over(state):
        return retry_game_over(game, reader, state, menu)
    if (state["screen"] == 7 and state["stage"] == 1
            and (game_over(state) or state["state"] == 0 and not state["continues"])):
        route, opening = full_credit_reset._recover_menu(game, reader, state, 30.)
        return dict(route=route, before=state, opening=opening)
    if restart_hook is not None:
        # The hook owns an operator-qualified, same-process route. Its output alone is never accepted.
        game.release_all_keys()
        game.resume_game_process()
        restart_hook(game, reader, state)
        opening = full_credit_reset._native_state(game, reader)
        full_credit_reset._require_fresh(opening)
        return dict(route="externally_qualified_restart", before=state, opening=opening)
    raise FreshCreditRequired(state, "Prepare a fresh Normal Reimu A credit or supply a qualified restart route.")


def reset(env, trial, *, start_frame, menu, restart_hook=None):
    state = full_credit_reset._native_state(env.game, env.reader)
    # A target-stage marker can appear before the next stage has become active.
    active_or_stage_transition = state["screen"] == 7 and state["state"] in (0, 2) and not state["continues"]
    route = None
    if not active_or_stage_transition and not (menu is None and game_over(state) and state["stage"] == 1):
        env.wait_for_focus()
        env.release()
        route = prepare(env.game, env.reader, menu, restart_hook)
        env.freeze()
    observation, info = full_credit_reset.fresh_credit_reset(env, trial, start_frame=start_frame)
    if route is not None:
        info["post_story_restart"] = route
    return observation, info
