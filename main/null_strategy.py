from qtpylib.algo import Algo

class NullAlgo(Algo):
    pass


if __name__ == "__main__":
    strategy = NullAlgo(
        instruments=[("SPCE", "OPT", "EMERALD", "USD", '20220121', 42.0, "CALL")],
        resolution="10T",
        tick_window=1000,
        preload="20T",
        ibport=7497
    )