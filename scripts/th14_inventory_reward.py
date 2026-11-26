"""Original TH14 1.00b Normal inventory potential, in spare-life and Bomb units."""
from struct import Struct

STOCK = Struct("<ii4xii")
LIFE_VALUE = .06
BOMB_VALUE = .012


def read_inventory(game):
    offsets = [game._OFFSETS[name] for name in ("lives","life_fragments","bombs","bomb_fragments")]
    if [address-offsets[0] for address in offsets] != [0,4,12,16]:
        raise ValueError("The TH14 original inventory offsets differ from the verified layout.")
    return STOCK.unpack(game._read_game_memory(offsets[0],STOCK.size))


def potential(stock):
    lives, life_pieces, bombs, bomb_pieces = stock
    if not (0 <= lives <= 8 and 0 <= life_pieces < 3 and 0 <= bombs <= 8 and 0 <= bomb_pieces < 8):
        raise ValueError(f"Invalid live TH14 inventory {stock!r}.")
    return LIFE_VALUE*(lives+life_pieces/3) + BOMB_VALUE*(bombs+bomb_pieces/8)


class InventoryReward:
    """At gamma=1 the shaping sum is zero on every completed fresh credit."""
    def __init__(self, initial):
        self.initial=potential(initial)
        self.previous=0.
        self.total=0.
        self.positive=0.
        self.negative=0.
        self.changes=0

    def advance(self, stock, *, terminal):
        current=0. if terminal else potential(stock)-self.initial
        change=current-self.previous
        self.total+=change
        self.positive+=max(0.,change)
        self.negative+=min(0.,change)
        self.changes+=abs(change)>1e-12
        self.previous=current
        if terminal and abs(self.total)>1e-7:
            raise ValueError("Terminal resource shaping did not telescope.")
        return change

    def report(self):
        return dict(total=self.total,positive=self.positive,negative=self.negative,events=self.changes)
