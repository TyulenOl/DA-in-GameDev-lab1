# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #4 выполнил(а):
- Тюленев Сергей Николаевич
- РИ210912

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |


Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Цель работы
### Узнать что такое перцептрон и ознакомиться с тем, как он работет.

## Задание 1
### В проекте Unity реализовать перцептрон, который умеет производить вычисления:
* ### OR | дать комментарии о корректности работы
* ### AND | дать комментарии о корректности работы
* ### NAND | дать комментарии о корректности работы
* ### XOR | дать комментарии о корректности работы
---
### Ход работы:
### Код скрипта отвечающий за работу перцептрона ###

```csharp
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

[System.Serializable]
public class TrainingSet
{
	public double[] input;
	public double output;
	
	public TrainingSet(double[] input, double output)
	{
		this.input = input;
		this.output = output;
	}
}

public class Perceptron : MonoBehaviour {

	public TrainingSet[] ts;
	double[] weights = {0,0};
	double bias = 0;
	double totalError = 0;

	double DotProductBias(double[] v1, double[] v2) 
	{
		if (v1 == null || v2 == null)
			return -1;
	 
		if (v1.Length != v2.Length)
			return -1;
	 
		double d = 0;
		for (int x = 0; x < v1.Length; x++)
		{
			d += v1[x] * v2[x];
		}

		d += bias;
	 
		return d;
	}

	double CalcOutput(int i)
	{
		double dp = DotProductBias(weights,ts[i].input);
		if(dp > 0) return(1);
		return (0);
	}

	void InitialiseWeights()
	{
		for(int i = 0; i < weights.Length; i++)
		{
			weights[i] = Random.Range(-1.0f,1.0f);
		}
		bias = Random.Range(-1.0f,1.0f);
	}

	void UpdateWeights(int j)
	{
		double error = ts[j].output - CalcOutput(j);
		totalError += Mathf.Abs((float)error);
		for(int i = 0; i < weights.Length; i++)
		{			
			weights[i] += error*ts[j].input[i]; 
		}
		bias += error;
	}

	public double CalcOutput(double i1, double i2)
	{
		double[] inp = new double[] {i1, i2};
		double dp = DotProductBias(weights,inp);
		if(dp > 0) return(1);
		return (0);
	}

	public void Train(int epochs)
	{
		InitialiseWeights();
		
		for(int e = 0; e < epochs; e++)
		{
			totalError = 0;
			for(int t = 0; t < ts.Length; t++)
			{
				UpdateWeights(t);
				Debug.Log("W1: " + (weights[0]) + " W2: " + (weights[1]) + " B: " + bias);
			}
			Debug.Log("TOTAL ERROR: " + totalError);
			if (totalError == 0)
			{
				Debug.Log($"Number of epochs before successful learning: {e + 1}".ToUpper());
				break;
			}
			if (e + 1 == epochs)
				Debug.Log($"Number of epochs before successful learning: {e + 1}".ToUpper());
		}
	}

	void Start ()
	{
		Train(10);
		Debug.Log("Test 0 0: " + CalcOutput(0,0));
		Debug.Log("Test 0 1: " + CalcOutput(0,1));
		Debug.Log("Test 1 0: " + CalcOutput(1,0));
		Debug.Log("Test 1 1: " + CalcOutput(1,1));		
	}
}

```

### Прикрепив данный скрипт в качестве компонента на пустой объект и подставляя разные наборы данных получаем:
1. #### Для логического оператора OR:<br /> 
	+ #### Запустив программу несколько раз можно сказать, что обучение в среднем происходит за 3-4 эпохи. Также хотелось бы отметить, что для оператора OR в среднем обучение происходит быстее чем у других логических операторов. ####  
	+ ![OR](https://user-images.githubusercontent.com/100992984/204286013-57ac49a1-fa6a-46f8-a3e4-f1e5531e6246.png)
	+ ![OR3](https://user-images.githubusercontent.com/100992984/204302237-a7158457-948d-4f2b-a48b-521cb9b80a42.png)
	+ ![OR4](https://user-images.githubusercontent.com/100992984/204302272-31923444-6611-46a1-8eb8-cae2e2d6b8fd.png)
	+ ![OR5](https://user-images.githubusercontent.com/100992984/204302282-05d97633-0391-4b89-a077-b799ecf0f909.png)

 
2. #### Для логического оператора AND:<br />
	+ #### Для данного оператора обучение перцептронна проходило в среднем за 6-7 эпох. ####
	+ ![AND](https://user-images.githubusercontent.com/100992984/204287036-d91e7b8e-8061-434f-b4c3-715eb5d6ab7c.png)
	+ ![AND1](https://user-images.githubusercontent.com/100992984/204302498-645a4759-8977-4977-b7fa-4b2e4545828a.png)
	+ ![AND2](https://user-images.githubusercontent.com/100992984/204302508-e6683b15-d45c-457a-a61b-198bc4c1fa0d.png)
	+ ![AND3](https://user-images.githubusercontent.com/100992984/204302628-1a8ccbc5-934e-47d8-8081-e2aade2de526.png)


3. #### Для логического оператора NAND:<br />
	+ #### Обучение происходит примерно за такое же количество эпох как и с оператором AND. ####
	+ ![NAND](https://user-images.githubusercontent.com/100992984/204301920-0d2054de-1afd-4657-97c5-3c44b8d09b02.png)
	+ ![NAND4](https://user-images.githubusercontent.com/100992984/204302769-f55b89ce-fdea-49e0-95a9-26b7eeb73e3c.png)
	+ ![NAND5](https://user-images.githubusercontent.com/100992984/204302784-dcb7756b-0864-47c6-80b3-b73a18152d03.png)
	+ ![NAND6](https://user-images.githubusercontent.com/100992984/204302795-fb7364b4-9c41-4b3c-864b-f939339ef914.png)


3. #### Для логического оператора XOR:<br />
	+ #### Не проходит обучение даже после 100 и 500 эпох. Это связано с тем что однослойные перцептроны могут работать только с линейно разделимыми данными. В нашем случае задача XOR является линейно неразделимой, а это значит, что перцептрон не может её решить. ####
	+ ![XOR](https://user-images.githubusercontent.com/100992984/204302968-6135af7e-6e38-4351-8ca5-0bab5450d79d.png)
	+ ![XOR1](https://user-images.githubusercontent.com/100992984/204303024-ed1711bb-f776-4dfe-952d-3b72f7e7c396.png)
	+ ![XOR2](https://user-images.githubusercontent.com/100992984/204303026-64deff57-3783-4b1a-8149-74a6886d4234.png)
	+ ![XOR3](https://user-images.githubusercontent.com/100992984/204303034-253e507a-d3c1-4cf3-9e29-d188e556424d.png)


## Задание 2
### Построить графики зависимости количества эпох от ошибки обучения. Указать от чего зависит необходимое количество эпох обучения.
---
### Ход работы:
### Для построения графиков я написал скрипт, который после выполнения генерирует csv файл с необходимыми данными.

```csharp
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class StatisticsCSV : MonoBehaviour
{
    private string fileName = "";

    private TrainingSet[] tsOR =
    {
        new(new double[] {0, 0}, 0),
        new(new double[] {1, 0}, 1),
        new(new double[] {0, 1}, 1),
        new(new double[] {1, 1}, 1),
    };
    private TrainingSet[] tsAND = 
    {
        new(new double[] {0, 0}, 0),
        new(new double[] {1, 0}, 0),
        new(new double[] {0, 1}, 0),
        new(new double[] {1, 1}, 1),
    };
    private TrainingSet[] tsNAND =
    {
        new(new double[] {0, 0}, 1),
        new(new double[] {1, 0}, 1),
        new(new double[] {0, 1}, 1),
        new(new double[] {1, 1}, 0),
    };
    private TrainingSet[] tsXOR =
    {
        new(new double[] {0, 0}, 0),
        new(new double[] {1, 0}, 1),
        new(new double[] {0, 1}, 1),
        new(new double[] {1, 1}, 0),
    };
    
    [SerializeField]private Perceptron Perceptron;

    private int numberEpochs = 10;
    private int repetitionСount = 100;
    
    void Start()
    {
        fileName = $"{Application.dataPath}/statistic.csv";
        var listOR = Statistic(tsOR);
        var listAND = Statistic(tsAND);
        var listNAND = Statistic(tsNAND);
        var listXOR = Statistic(tsXOR);
        WriteCSV(new[] {listOR, listAND, listNAND, listXOR});
    }

    private List<double> Statistic(TrainingSet[] ts)
    {
        var listReceivedValues = new List<double>[numberEpochs];

        for (var i = 0; i < numberEpochs; i++)
            listReceivedValues[i] = new List<double>();

        for(var i = 0; i < repetitionСount; i++)
        {
            var listTotalErrors = Perceptron.Train(numberEpochs, ts);
            for (var j = 0; j < listTotalErrors.Count; j++)
            {
                listReceivedValues[j].Add(listTotalErrors[j]);
            }
        }

        var averageList = new List<double>();
        foreach (var value in listReceivedValues)
        {
            averageList.Add(value.Average());
        }

        return averageList;
    }

    private void WriteCSV(List<double>[] array)
    {
        var textWriter = new StreamWriter(fileName, false);
        textWriter.WriteLine("Epoch number; OR; AND; NAND; XOR");
        textWriter.Close();
        
        textWriter = new StreamWriter(fileName, true);

        for (int i = 0; i < numberEpochs; i++)
        {
            textWriter.WriteLine($"{i + 1}; {array[0][i]:0.00}; {array[1][i]:0.00}; {array[2][i]:0.00}; {array[3][i]:0.00}");
        }
        textWriter.Close();
    }
}
```
### Данный скрипт запускает обучение перцептрона 100 раз для каждого логического оператора и из полученных результатов вычисляет средее арифметическое количества ошибок для каждой эпохи. Из полученных данных генерируется csv фаил на основе которых можно построить необходимые графики.<br />

### Полученные данные:
![table](https://user-images.githubusercontent.com/100992984/204309531-2a0f49a4-f6fa-48ef-94b9-5a2cb6034d2d.png)

### Графики:
* OR
![graphOR](https://user-images.githubusercontent.com/100992984/204312514-cf7b2088-f7e7-41e5-af22-92a46f04c761.png)


* AND
![graphAND](https://user-images.githubusercontent.com/100992984/204313022-993f4878-3312-4cd7-b6d2-76e605b0e74c.png)


* NAND
![graphNAND](https://user-images.githubusercontent.com/100992984/204313039-679d6f4b-b70e-4606-ada7-eea02bf141f1.png)


* XOR
![graphXOR](https://user-images.githubusercontent.com/100992984/204313048-1c3e5fd2-3306-40bb-add3-1e216a050c49.png)




## Задание 3
### Изучить код на Python и ответить на вопросы:
### 1. Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.
Величина loss будет равна нулю, если исходные данные x и y будут заданы линейной функцией и коэффициент b = 0
![Screenshot_13](https://user-images.githubusercontent.com/100992984/191829750-eaeb35b6-e298-4cc4-907b-834148887ec9.png)


### 2. Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.
Параметр Lr отвечает за величину шага, от которой зависит, насколько быстро или медленно будет происходить достижение оптимальных значений. 
Чем больше параметр Lr, тем быстее будет происходить возрастание графика.
Так, при Lr=0.0002 график за 10 повторений достигает примерно тех же значений, что и при Lr=0.000001 за 10000 повторений.
![Screenshot_11](https://user-images.githubusercontent.com/100992984/191812930-d7bdd33b-474d-4500-bea5-cbe859c0f608.png)
![Screenshot_12](https://user-images.githubusercontent.com/100992984/191812939-5be2cda7-265f-4694-96d5-1a5721b258fa.png)

При очень маленьких значения Lr можно получить более точный график, но он будет расти медленнее и следовательно, потребуется больше времени для получения оптимального решения. При слишком больших значениях Lr график может и вовсе пройти мимо минимального значения.

## Выводы

В ходе лабораторной работы:
- Установил и запустил програмное обеспечение для работы с Python, а таже Unity.
- Написал маленькие программы по выводу Hello world в Google.Colab и в консоль Unity.
- Ознакомился с работой на Python на примере алгоритма линейной регрессии.
- Рассмотрел работу линейной регрессии.
