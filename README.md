# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
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
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.

### Ход работы:
- Создание нового 3D проекта на юнити:
![Screenshot_1](https://user-images.githubusercontent.com/100992984/197847376-ed839f31-ee20-4ad8-b584-281e2960a826.png)


- Добавление необходимих пакетов в проект Unity для работы с MLAgent:
![Screenshot_2](https://user-images.githubusercontent.com/100992984/197847564-c11c82f6-4ce7-4b80-ab43-f7141917a417.png)


- Cоздание и активации нового ML-агента, а также скачивание необходимых библиотек:
![Screenshot_3](https://user-images.githubusercontent.com/100992984/197847897-c5237dfd-9c1d-49bc-b88d-9a021af62939.png)


- Создание плоскости, куба, сферы и скрипта для MLAgent'а:
![Screenshot_4](https://user-images.githubusercontent.com/100992984/197849546-14a1bb26-1bcc-4d2a-a664-fd75fbfdb27a.png)


- Код содержащийся в скрипт-файле RollerAgent.cs
```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```


- Добавление необходимых компонетов для обекта сферы:             
![Screenshot_9](https://user-images.githubusercontent.com/100992984/197850341-ffab02cf-d48e-456e-8b7d-ade233d51900.png)


- Обучение с 1 экземпляром агента:
![Screenshot_6](https://user-images.githubusercontent.com/100992984/197850962-2d49bd96-c318-4bbc-a078-6059cd0da60a.png)


- Обучение с 9-ю экземплярами агента:
![Screenshot_7](https://user-images.githubusercontent.com/100992984/197851152-b34da8a5-587d-4ec5-849e-5c72596ce254.png)


- Обучение с 36-ю экземплярами агента:
![Screenshot_8](https://user-images.githubusercontent.com/100992984/197851249-87569735-f1cd-4dd8-a90b-8a26546fa082.png)


- Результат работы модели после обучения:


https://user-images.githubusercontent.com/100992984/197851790-3f7d1d8b-c604-40e5-b9d6-43b82f3e34f3.mp4


### Исходя из полученных результатов можно сделать вывод о том, что количество экземпляров агента влияют на скорость обучения модели. В нашем случае, когда экземпляр агента был всего один, то обучение происходило довольно медленно, но по мере увеличения количества обучаемых агентов скорость обучения тоже возрастала. Это связано с тем, что при использовании нескольких агентов, за одну итерацию собирается большее количество данных. Следовательно, чем больше количество экземпляров агента, тем быстрее происходит процесс обучения.



## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере. 

### Ход работы:

- Описание работы строк файла конфигурации нейронной сети:
```yaml
behaviors:
  RollerBall:
    trainer_type: ppo # Используемый тип тренажера. ppo - Это алгоритм обучения с подкреплением на основе политики.
    hyperparameters:
      batch_size: 10 # Количество опытов в каждой итерации градиентного спуска. Всегда должно быть в несколько раз меньше, чем buffer_size.
      buffer_size: 100 # Количество опытов, которые необходимо собрать перед обновлением модели политики. Соответствует тому, сколько опыта должно быть собрано, прежде чем мы будем изучать или обновлять модель. Значение должно быть в несколько раз больше, чем batch_size.
      learning_rate: 3.0e-4 # Начальная скорость обучения для градиентного спуска. Соответствует силе каждого шага обновления градиентного спуска.
      beta: 5.0e-4 # Сила энтропийной регуляризации, которая делает политику «более случайной». Это гарантирует, что агенты должным образом исследуют пространство действия во время обучения. 
      epsilon: 0.2 # Влияет на то, насколько быстро политика может развиваться во время обучения. Соответствует допустимому порогу расхождения между старой и новой политикой при обновлении градиентного спуска.
      lambd: 0.99 # Параметр регуляризации, используемый при расчете обобщенной оценки преимущества. Это можно рассматривать как то, насколько агент полагается на свою текущую оценку стоимости при вычислении обновленной оценки стоимости.
      num_epoch: 3 # Количество проходов через буфер опыта при оптимизации градиентного спуска. Уменьшение этого параметра обеспечит более стабильные обновления за счет более медленного обучения.
      learning_rate_schedule: linear # Определяет, как скорость обучения изменяется с течением времени. linear - уменьшает learning_rate линейно, достигая 0 при max_steps
    network_settings:
      normalize: false # Определяет применяется ли нормализация к входам векторного наблюдения. Эта нормализация основана на скользящем среднем и дисперсии векторного наблюдения. 
      hidden_units: 128 # Количество блоков в скрытых слоях нейронной сети. Соответствуют количеству единиц в каждом полносвязном слое нейронной сети
      num_layers: 2 # Количество скрытых слоев в нейронной сети. Соответствует количеству скрытых слоев после ввода наблюдения или после кодирования визуального наблюдения сверточных нейронных сетей.
    reward_signals: # Позволяет задавать настройки как для внешних, так и для внутренних сигналов вознаграждения
      extrinsic:
        gamma: 0.99 # Коэффициент дисконтирования для будущих вознаграждений, поступающих из среды.
        strength: 1.0 # Коэффициент, на который умножается вознаграждение, выдаваемое средой. Типичные диапазоны зависят от сигнала вознаграждения.
    max_steps: 500000 # Общее количество шагов, которые должны быть выполнены в среде до завершения процесса обучения. Если в среде имеется несколько агентов с одинаковым именем поведения, все шаги, предпринятые этими агентами, будут способствовать подсчету max_steps.
    time_horizon: 64 # Отвечает за количество шагов опыта, которые нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта. Когда этот предел достигается до конца эпизода, используется оценка стоимости для прогнозирования общего ожидаемого вознаграждения от текущего состояния агента. 
    summary_freq: 10000 # Количество опытов, которое необходимо собрать перед генерацией и отображением статистики обучения.
```


Компонент DecisionRequester автоматически запрашивает решения для экземпляра агента через регулярные промежутки времени.

Компонент BehaviorParameters предназначен для настройки поведения экземпляра агента и свойств мозга.



## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

### Ход работы:

- Доработанный код для нахождения и перемещения агента до точки между двумя разноцветными кубами:
```c#
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Serialization;
using Random = UnityEngine.Random;

public class RollerAgent : Agent
{
    private Rigidbody rBody;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform greenTarget;
    public Transform blueTarget;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }


        greenTarget.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        blueTarget.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var pointBetweenTarget = (greenTarget.localPosition + blueTarget.localPosition) / 2;
        sensor.AddObservation(pointBetweenTarget);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionsBuffers)
    {
        var controlSignal = Vector3.zero;
        controlSignal.x = actionsBuffers.ContinuousActions[0];
        controlSignal.z = actionsBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        var pointBetweenTarget = (greenTarget.localPosition + blueTarget.localPosition) / 2;
        var distanceToTarget = Vector3.Distance(this.transform.localPosition, pointBetweenTarget);

        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```


- Обучение агента для работы с двумя кубами:
![Screenshot_10](https://user-images.githubusercontent.com/100992984/197856285-01e22ba2-c77b-4904-ac0b-5e84565c591e.png)


- Результат работы модели с двумя разноцветными кубами после обучения:


https://user-images.githubusercontent.com/100992984/197856566-042eb2af-6f7d-40c3-ab6a-78d7017c4282.mp4





## Выводы

### Что такое игровой баланс?
- Игровой баланс - это состояние игры, когда все механики, характеристики, способности предметов, объектов, существ находятся примерно на одном уровне и среди них нет таких, которые могли бы сломать игровой процесс, сделав его слишком лёгким или сложным. Достичь идеального игрового баланса невозможно, но при разработке игры разработчики могут постараться добиться такого уровня баланса, при котором игроку не будет скучно играть из-за слишком сильных предметов или неинтересно из-за слишком сильных врагов и непроходимых боссов.

### Как системы машинного обучения могут быть использованы в игровом балансе?
- Я думаю, что специально обученные системы машинного обучения могут быть использованы в игровом балансе, например, для того чтобы на основе характеристик и скорости развития персонажа игрока подгонять уровень сложности врагов делая игровой процесс более плавным и приятным.
