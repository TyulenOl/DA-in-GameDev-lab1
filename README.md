# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #5 выполнил(а):
- Тюленев Сергей Николаевич
- РИ210912

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 80 |
| Задание 2 | * | 20 |


Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Цель работы
### Интеграция экономической системы в проект Unity и обучение ML-Agent.

## Задание 1
###  Измените параметры файла. yaml-агента и определить какие параметры и как влияют на обучение модели.
---
### Ход работы:
### Установка необходимых компонетнов для работы с ML-агентом ###
![3](https://user-images.githubusercontent.com/100992984/205153159-d17612cb-5d86-419c-8622-1c03aa4358d8.png)
![4](https://user-images.githubusercontent.com/100992984/205153307-98140c5e-95c8-444b-a351-489bd66ca606.png)
![5](https://user-images.githubusercontent.com/100992984/205153320-dbb549f1-4ed3-4023-8cfb-1624994cf773.png)

### Перед запуском обучения агента решил изменить максимальное количество шагов(max_steps) с 750000 до 100000.
![123](https://user-images.githubusercontent.com/100992984/205154820-8153a4c0-f6a6-42b1-85a6-2b9751ebc914.png)

### Начало и результат обучения предложенной модели.
![6](https://user-images.githubusercontent.com/100992984/205154970-33413061-968d-44c2-951e-cb53d176176e.png)
![economic_res](https://user-images.githubusercontent.com/100992984/205155040-89f87941-7b41-46d5-b937-2d1a46d26258.png)

### Установка TensorBoard для посторения графиков, чтобы впоследстии оценивать результаты обучения.
![8](https://user-images.githubusercontent.com/100992984/205155569-172c1999-1138-4e27-9497-672adb05c531.png)

### После запуска TensorBoard получаем следующие графики:
![ec1_1](https://user-images.githubusercontent.com/100992984/205262021-86335696-a8bf-4cfd-baca-768de96df59b.png)

### Изменим некоторые значения из .yaml файла для того, чтобы увидеть какие параметры и как влияют на обучение.
Начальные значения .yaml файла и то, за что они отвечают:
```yaml
behaviors:
  Economic:
    trainer_type: ppo # Используемый тип тренажера. ppo - Это алгоритм обучения с подкреплением на основе политики.
    hyperparameters:
      batch_size: 1024 # Количество опытов в каждой итерации градиентного спуска. Всегда должно быть в несколько раз меньше, чем buffer_size.
      buffer_size: 10240 # Количество опытов, которые необходимо собрать перед обновлением модели политики. Соответствует тому, сколько опыта должно быть собрано, прежде чем мы будем изучать или обновлять модель. Значение должно быть в несколько раз больше, чем batch_size.
      learning_rate: 3.0e-4 # Начальная скорость обучения для градиентного спуска. Соответствует силе каждого шага обновления градиентного спуска.
      learning_rate_schedule: linear # Определяет, как скорость обучения изменяется с течением времени. linear - уменьшает learning_rate линейно, достигая 0 при max_steps
      beta: 1.0e-2 # Сила энтропийной регуляризации, которая делает политику «более случайной». Это гарантирует, что агенты должным образом исследуют пространство действия во время обучения.
      epsilon: 0.2 # Влияет на то, насколько быстро политика может развиваться во время обучения. Соответствует допустимому порогу расхождения между старой и новой политикой при обновлении градиентного спуска.
      lambd: 0.95 # Параметр регуляризации, используемый при расчете обобщенной оценки преимущества. Это можно рассматривать как то, насколько агент полагается на свою текущую оценку стоимости при вычислении обновленной оценки стоимости.
      num_epoch: 3 # Количество проходов через буфер опыта при оптимизации градиентного спуска. Уменьшение этого параметра обеспечит более стабильные обновления за счет более медленного обучения.  
    network_settings:
      normalize: false # Определяет применяется ли нормализация к входам векторного наблюдения. Эта нормализация основана на скользящем среднем и дисперсии векторного наблюдения.
      hidden_units: 128 # Количество блоков в скрытых слоях нейронной сети. Соответствуют количеству единиц в каждом полносвязном слое нейронной сети
      num_layers: 2 # Количество скрытых слоев в нейронной сети. Соответствует количеству скрытых слоев после ввода наблюдения или после кодирования визуального наблюдения сверточных нейронных сетей.
    reward_signals:
      extrinsic:
        gamma: 0.99 # Коэффициент дисконтирования для будущих вознаграждений, поступающих из среды.
        strength: 1.0 # Коэффициент, на который умножается вознаграждение, выдаваемое средой. Типичные диапазоны зависят от сигнала вознаграждения.
    checkpoint_interval: 500000 # Количество опыта, полученного тренером между каждой контрольной точкой.
    max_steps: 100000 # Общее количество шагов, которые должны быть выполнены в среде до завершения процесса обучения. Если в среде имеется несколько агентов с одинаковым именем поведения, все шаги, предпринятые этими агентами, будут способствовать подсчету max_steps.
    time_horizon: 64 # Отвечает за количество шагов опыта, которые нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта. Когда этот предел достигается до конца эпизода, используется оценка стоимости для прогнозирования общего ожидаемого вознаграждения от текущего состояния агента.
    summary_freq: 5000 # Количество опытов, которое необходимо собрать перед генерацией и отображением статистики обучения.
    self_play:
      save_steps: 20000 # Количество шагов обучения между снапшотами
      team_change: 100000 # Количество trainer_steps между переключением учебной группы.
      swap_steps: 10000 # Количество призрачных шагов (не шагов тренера) между сменой политики оппонентов на другой снапшот. Под "призрачным шагом" понимается шаг, сделанный агентом, который следует фиксированной политике и не обучается.
      play_against_latest_model_ratio: 0.5 # Вероятность того, что агент будет играть против последней политики противника.
      window: 10 # Размер выборки прошлых снапшотов, из которых выбираются противники агента.
```

1. ### Для начала я решил изменить значения learning_rate: 3.0e-4 --> 1.0e-3
+ #### После обучения получаем следующие графики:
+ ![ec_lr-3](https://user-images.githubusercontent.com/100992984/205271277-114c75d6-8f7b-4857-8896-4749ff3aa63c.png)
+ Синий график - без изменений, оранжевый график - с изменениями.
+ ![ec_lr-1](https://user-images.githubusercontent.com/100992984/205270474-eff759fa-7793-42cb-b39e-19078283a517.png)
+ #### График Cumulative Reward(общее вознаграждение): ####
* После увеличения параметра learning_rate изменение значений графика стали более случайными и резкими. Это привело к тому что в конце обучения общее вознаграждение оказалось примерно равными начальному значению в то время, как график агента без изменения на протяжении всего обучения в основном возрастает. 
+ #### График Policy Loss: ####
* Оранжевый график Policy Loss изначально имеет довольно большие значения, но по мере обучения потери принятия решений уменьшаются и в конце обучения становятся даже немного меньше, чем у синего графика.
+ #### График Value Loss: ####
* Почти на протяжении всего обучения потери значений агента с изменениями меньше чем у агента без изменений.


### Прикрепив данный скрипт в качестве компонента на пустой объект и подставляя разные наборы данных получаем:
1. #### Для логического оператора OR:<br /> 
	+ #### Запустив программу несколько раз можно сказать, что обучение в среднем происходит за 3-4 эпохи. Также хотелось бы отметить, что для оператора OR в среднем обучение происходит быстрее чем у других логических операторов. ####  
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
### Данный скрипт запускает обучение перцептрона 100 раз для каждого логического оператора и из полученных результатов вычисляет среднее арифметическое количества ошибок для каждой эпохи. Из полученных данных генерируется csv файл на основе которых можно построить необходимые графики.<br />

### Полученные данные:
![table](https://user-images.githubusercontent.com/100992984/204309531-2a0f49a4-f6fa-48ef-94b9-5a2cb6034d2d.png)

### Графики:
* OR: <br />
![graphOR](https://user-images.githubusercontent.com/100992984/204314116-2498a890-d192-40a3-becd-3736e79f3487.png)

* AND: <br />
![graphAND](https://user-images.githubusercontent.com/100992984/204314137-64bf015f-4f6a-4036-bc9d-7b28eb7347ab.png)

* NAND: <br />
![graphNAND](https://user-images.githubusercontent.com/100992984/204314155-74c3c745-cd90-4e2d-b49c-1c4abe099f07.png)

* XOR: <br />
![graphXOR](https://user-images.githubusercontent.com/100992984/204314160-cc42cb12-d198-4473-a567-a252da52c63d.png)

### Количество эпох обучения зависит от сложности операции, которую перцептрон пытается научиться решать, а также от того насколько удачно изначально были подобранны веса и смещение.


## Задание 3
### Построить визуальную модель работы перцептрона на сцене Unity.
---
### Ход работы:

#### Для визуализации работы перцептрона я решил сделать 8 кубиков, которые падают друг на друга и при соприкосновении создают новый кубик с необходимым значением.

#### Для начала на сцене я создал 8 кубиков с прикреплённым на них текстом и платформу, на которую кубы будут падать.
![scr1](https://user-images.githubusercontent.com/100992984/204333764-cff2c7a1-7778-4686-8d66-e4566fc39de8.png)

#### К каждому кубику я прикрепил компонент Rigidbody для использования физики, а также скрипт CubeData. Скрипт CubeData содержит метод OnTriggerEnter для обработки столкновения и метод ResetCube для сброса состояния куба. Метод OnTriggerEnter меняет соответствующим образом взаимодействующие кубы и передает необходимые данные в контроллер.
```csharp
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class CubeData : MonoBehaviour
{
    public double value;
    public Material cubeColor;
    public Color textColor;

    private Vector3 initialPosition;

    private void Start()
    {
        gameObject.name = value.ToString();
        gameObject.GetComponent<MeshRenderer>().material = cubeColor;
        GetComponentInChildren<TextMeshPro>().SetText(value.ToString());
        textColor.a = 1;
        GetComponentInChildren<TextMeshPro>().color = textColor;
        initialPosition = transform.position;
    }

    private void OnTriggerEnter(Collider other)
    {
        other.gameObject.SetActive(false);
        gameObject.SetActive(false);
        var valueCube1 = value;
        var valueCube2 = other.gameObject.GetComponent<CubeData>().value;
        if(gameObject.GetComponent<BoxCollider>().isTrigger)
            Controller.Instance.HandleCollision(other.transform.position, valueCube1, valueCube2);
    }

    public void ResetCube()
    {
        this.gameObject.SetActive(true);
        transform.position = initialPosition;
    }
}
```
![scr2](https://user-images.githubusercontent.com/100992984/204350988-9cfcda90-9541-4ad0-8746-dd70e7222b24.png)

#### Далее я создал кнопки для смены набора обучающих данных, что бы можно было менять логику поведения перцептрона.
![scr3](https://user-images.githubusercontent.com/100992984/204352390-537968fe-2359-450a-ba7b-ad2d01d034d2.png)

#### На каждую кнопку повесил скрипт ButtonChangeSet. Благодаря данному скрипту в кнопке можно хранить массив обучающих данных и при клике на кнопку передавать их в контроллер.
```csharp
using UnityEngine;

public class ButtonChangeSet : MonoBehaviour
{
    [SerializeField] private TrainingSet[] ts;

    public void ChangeSet()
    {
        Controller.Instance.OnButtonClick(ts);
    }
}
```

#### В свою очередь, контроллер - это пустой объект на который прикреплены скрипты Controller и Perceptron. 
![scr4](https://user-images.githubusercontent.com/100992984/204353744-5ef3f013-0815-4ff5-a59a-97e94a16fcc0.png)

#### Скрипт Controller принимает данные от кнопок и переобучает перцептрон. Также контроллер при получении данных от столкнувшихся кубов обрабатывает их значения при помощи перцептрона и на основе результата создаёт соответствующий значению куб. 
```csharp
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class Controller : MonoBehaviour
{
    [SerializeField] private Perceptron perceptron;

    private static Controller instance;

    public static Controller Instance
    {
        get
        {
            if(instance == null)
            {
                instance = FindObjectOfType(typeof(Controller)) as Controller;
                if(instance == null)
                {
                    instance = new GameObject("Controller").AddComponent<Controller>();
                }
            }
            return instance;
        }
    }

    private GameObject[] cubes;
    private List<GameObject> tempCubes = new List<GameObject>();

    [SerializeField] private GameObject whiteCube;
    [SerializeField] private GameObject blackCube;

    private bool isActive;

    private void Start()
    {
        cubes = GameObject.FindGameObjectsWithTag("Cube");
    }

    public void OnButtonClick(TrainingSet[] ts)
    {
        if(isActive)
            return;
        
        isActive = true;
        perceptron.Train(10, ts);
        foreach (var cube in cubes)
        {
            cube.gameObject.GetComponent<Rigidbody>().useGravity = true;
        }
    }

    public void ResetCubes()
    {
        foreach (var cube in cubes)
        {
            cube.gameObject.GetComponent<Rigidbody>().useGravity = false;
            cube.gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            cube.GetComponent<CubeData>().ResetCube();
        }

        foreach (var tempCube in tempCubes)
        {
            Destroy(tempCube);
        }
        tempCubes.Clear();

        isActive = false;
    }

    public void HandleCollision(Vector3 position, double value1, double value2)
    {
        var output = perceptron.CalcOutput(value1, value2);
        if (output == 0)
        {
            var tempCube = Instantiate(whiteCube, position, Quaternion.identity);
            tempCubes.Add(tempCube);
        }
        else if (output == 1)
        {
            var tempCube = Instantiate(blackCube, position, Quaternion.identity);
            tempCubes.Add(tempCube);
        }
    }
}
```

#### Помимо этого, на сцене присутствует кнопка reset. При нажатии на эту кнопку происходит вызов метода ResetCubes из скрипта Controller. Метод ResetCubes сбрасывает состояния всех кубов, возвращая их в начальное положение.
![scr5](https://user-images.githubusercontent.com/100992984/204357537-c44031b0-edcd-4ae4-8050-5678192e9f1a.png)

### Демонстрация работы визуальной модели работы перцептрона:


https://user-images.githubusercontent.com/100992984/204357828-b56d498c-b586-4230-8080-b918ff2a8479.mp4





## Выводы

В ходе лабораторной работы:
- Узнал, что такое перцептрон.
- Ознакомился с работой перцептрона.
- Построил графики зависимости количества эпох от ошибок обучения и сделал соответствующий вывод.
- Создал визуальную модель работы перцептрона.
