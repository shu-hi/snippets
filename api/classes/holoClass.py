from abc import ABC, abstractmethod
import func


envs = func.get_envs()


# Abstract Factoryパターン※=interface
class AbstructHoloClass(ABC):
    @abstractmethod
    def store(self):
        pass


# Simple Factoryパターン※分岐が隠蔽できるのを活用
class holoFactory:
    @staticmethod
    async def create(request):
        data = await request.json()
        if data["mediaType"] == "pic":
            material = {k: v for k, v in data.items() if k != "mediaType"}
            return picHoloClass(material)
        elif data["mediaType"] == "vid":
            return vidHoloClass(data["video"])
        else:
            raise ValueError("Unknown mediaType")


class holoPics:
    def __init__(self, up, right, down, left):
        self.__uppic = up
        self.__rightpic = right
        self.__downpic = down
        self.__leftpic = left

    @property
    def uppic(self):
        return self.__uppic

    @property
    def rightpic(self):
        return self.__rightpic

    @property
    def downpic(self):
        return self.__downpic

    @property
    def leftpic(self):
        return self.__leftpic


class picHoloClass(AbstructHoloClass):
    def __init__(self, holo_pics):
        super().__init__()
        self.holo_pics = holo_pics

    def store(self):
        pass


class vidHoloClass(AbstructHoloClass):
    def __init__(self, video):
        self.video = video

    def store(self):
        pass


# 表示側のクラス　strategyで作ってみる
class showHolo:
    _strategy = None  # privateです……

    def __init__(self, strategy):
        self._strategy = strategy

    def setStrategy(self, strategy):
        self._strategy = strategy

    def showMedia(self):
        self._strategy.show()
