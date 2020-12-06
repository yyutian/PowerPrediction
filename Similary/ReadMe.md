##函数介绍
Test_LSTM.py   --使用已训练好的模型进行预测核对结果的评估

ResultAnalysis.py--LSTM模型结果分析

Training_LSTM.py--训练模型

LinearRegression.py     --使用线性回归模型预测

RandomForest.py --使用随机森林预测

SimilaryFill.py---->Similary_Reverse_Fill.py--前后相似填充数据

SimilaryProcess_All.py--->MergeData.py--处理训练用到的所有数据，然后把数据整合在一起

SimilaryProcess.py--处理单个用来测试的电站数据集

Method.py--封装计算相似距离方法

Metrics.py--评价函数

Sure_Capacity.py    --确保自己获取的数据容量属性都是正确的(即不是自己填充的假数据)

ProcessWeather.py--把天气的名称转换为编码

SVM.py--支持向量机模型

##datasets介绍

FillData：填充的数据

FinalData：最终数据

MODEL存放训练好的模型

pic存放一些结果图

RET训练后的结果

#### <u>shanghe_weather.csv</u>
商河县2017-2020年的天气信息
#### <u>jinan_weather.csv</u>
济南整个区域的天气信息
#### <u>FillData文件夹</u>
使用相似日算法填充缺失的发电量数据
#### <u>shanghe_weather.csv</u>
商河详细的天气信息
#### <u>shanghe_weather_by_day.csv</u>

商河每天的天气信息，没有细分
        
#### <u>天气信息表</u>

路径：data/jinan_weather.csv

city_Code:城市编码

city_Name:城市名字

time:日期

weatherCode:天气编码

weatherTypeName:天气名称

humidity:湿度

temperature:温度

wind:风速加风向



#### <u>最终输入信息</u>

路径：data/FinalData/similary{}.csv

day_s{}:相似日日期

power_s{}:相似日发电量

time:预测日时间

dayPower:预测日发电量