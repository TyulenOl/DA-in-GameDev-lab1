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
* #### Код скрипта отвечающий за работу перцептрона

```c#
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

* #### Прикрепив данный скрипт в качестве компонента на пустой объект и подставляя разные наборы данных получаем:
 1. #### Для логического оператора OR:<br />
  Запустив программу несколько раз можно сказать, что обучение в среднем происходит за 3-4 прохода. Также хотелось бы отметить, что для оператора OR в среднем        обучение происходит быстее чем у других логических операторов.     

  ![OR](https://user-images.githubusercontent.com/100992984/204286013-57ac49a1-fa6a-46f8-a3e4-f1e5531e6246.png)
  ![OR2](https://user-images.githubusercontent.com/100992984/204286127-6de1e3e6-5b98-4f6d-9163-8c776fab0bb6.png)


 2. ##### Для логического оператора AND:<br />
  Для данного оператора обучение перцептронна проходило в среднем за 6-7 проходов.

  ![AND](https://user-images.githubusercontent.com/100992984/204287036-d91e7b8e-8061-434f-b4c3-715eb5d6ab7c.png)
  ![AND1](https://user-images.githubusercontent.com/100992984/204287114-a9cb4dee-b9a4-40e9-abce-5c0bcaafd1e7.png)
	







## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
### Ход работы:
* #### Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
In [ ]:
import numpy as np
import matplotlib.pyplot as plt

x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
x = np.array(x)
y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 199]
y = np.array(y)

plt.scatter(x, y)
plt.show()

```
![Screenshot_4](https://user-images.githubusercontent.com/100992984/191831208-37e7a64c-2165-4f77-8e7e-265043770c9f.png)


* #### Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
import numpy as np
import matplotlib.pyplot as plt

x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
x = np.array(x)
y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 199]
y = np.array(y)

plt.scatter(x, y)


def model(a, b, x):
    return a * x + b


def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5 / num) * (np.square(prediction - y)).sum()


def optimize(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * ((prediction - y).sum())
    a = a - Lr * da
    b = b - Lr * db
    return a, b


def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b
```
* #### Начать итерацию 
1. Инициализаци и модель итеративной оптимизации 
![Screenshot_5](https://user-images.githubusercontent.com/100992984/191805481-8576660f-dc87-4d64-942c-6025ebe22801.png)

2. На второй итерации отображаются значения параметров, значения потрерь и эффекты визуализации после итерации
![Screenshot_6](https://user-images.githubusercontent.com/100992984/191805510-be494e4e-0563-4b03-b4d9-4d42dbe5b96c.png)

3. Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации
![Screenshot_7](https://user-images.githubusercontent.com/100992984/191805530-b95bfcc6-023c-4bdb-b21d-bd9fad611a92.png)

4. На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
![Screenshot_8](https://user-images.githubusercontent.com/100992984/191805575-6e388906-f91a-46ff-a5ba-4000510b1fdd.png)

5. Пятая итерация показывает значения параметра, значение потреь и эффект визуализации после итерации
![Screenshot_9](https://user-images.githubusercontent.com/100992984/191805610-781b7ec1-1b6e-4374-bf50-53812128a61c.png)

6. 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
![Screenshot_10](https://user-images.githubusercontent.com/100992984/191805629-3dda2647-29d5-414e-87e2-00ecaae5dee6.png)


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
