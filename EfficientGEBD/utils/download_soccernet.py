import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/data1/socnet")
mySoccerNetDownloader.password = "s0cc3rn3t"
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) 