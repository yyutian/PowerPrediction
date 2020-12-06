import math
'''
   对天气、湿度、温度进行归一化
'''
class Method(object):
    def __init__(self,T_MAX,T_MIN,H_MAX,H_MIN,wcount):
        self.T_MAX=T_MAX
        self.T_MIN=T_MIN
        self.H_MAX=H_MAX
        self.H_MIN=H_MIN
        self.wcount=wcount

    def temperatureUniform(self,temperatureList):  # 温度归一化
        uniformTemperatureList = []
        for item in range(0, len(temperatureList)):
            uniformTemperatureList.append((self.T_MAX - temperatureList[item]) / (self.T_MAX - self.T_MIN))
        return uniformTemperatureList

    def humidityUniform(self,humidityList):  # 湿度归一化
        uniformHumidityList = []
        for item in range(0, len(humidityList)):
            uniformHumidityList.append((self.H_MAX - humidityList[item]) / (self.H_MAX - self.H_MIN))
        return uniformHumidityList

    def weatherUniform(self,weatherList):  # 天气归一化
        uniformWeatherList = []
        for item in range(0, len(weatherList)):
            uniformWeatherList.append(weatherList[item] / self.wcount)
        return uniformWeatherList

    '''
       获取预测日与相似日各参数的欧氏距离
       get
    '''
    def getDistance(self,similarParam,forecastParam):
        square = 0.0
        for i in range(0, len(forecastParam)):
            # （预测日 - 相似日）平方
            square += (similarParam[i] - forecastParam[i]) * (similarParam[i] - forecastParam[i])
            #print("预测日 - 相似日:",similarParam[i] - forecastParam[i])
        root = math.sqrt(square)#欧式距离公式
        #print(square)
        return root