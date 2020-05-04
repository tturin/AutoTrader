from qtpylib.blotter import Blotter

class MainBlotter(Blotter):
    pass


if __name__ == "__main__":
    blotter = MainBlotter(
        dbhost="localhost",
        dbuser="autotrader",
        dbpass="Ogaboga123!",
        ibport=7497
    )

    blotter.run()
