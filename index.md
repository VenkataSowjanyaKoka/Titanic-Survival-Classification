```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>



```python
#for structured data we use Spark SQL, SparkSession acts a pipeline between data and sql statements
from pyspark.sql import SparkSession
```


```python
# sparksession is like a class and we need to create an instance of a class to utilize
spark = SparkSession.builder.appName("Train_Data_Logistic_Regression").getOrCreate()
```


```python
#To assign dummy values to the string variables
from pyspark.ml.feature import StringIndexer
```


```python
#For creation of a vector of input variables
from pyspark.ml.feature import VectorAssembler
```


```python
#For assigning the dummy variables
from pyspark.ml.feature import OneHotEncoder
```


```python
#Loading the logistic regression model
from pyspark.ml.classification import LogisticRegression
```


```python
#Reading the csv file data
Train_DF = spark.read.csv("/Users/sowjanyakoka/Desktop/Spring2020/MachineLearning/Titanics.csv", inferSchema = True, header = True)
```


```python
#Seeing the shape of the dataset
print("Shape:", (Train_DF.count(), len(Train_DF.columns)))
```

    Shape: (891, 12)



```python
#Looking at the top 10 rows data of the dataset
Train_DF.show(10, truncate = True)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|
    |          6|       0|     3|    Moran, Mr. James|  male|null|    0|    0|          330877| 8.4583| null|       Q|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|           17463|51.8625|  E46|       S|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1|          349909| 21.075| null|       S|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| null|       S|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| null|       C|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 10 rows
    



```python
#Looking at the descriptive statistics of the dataset
Train_DF.describe().show(truncate = False)
```

    +-------+-----------------+-------------------+------------------+------------------------------------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
    |summary|PassengerId      |Survived           |Pclass            |Name                                            |Sex   |Age               |SibSp             |Parch              |Ticket            |Fare             |Cabin|Embarked|
    +-------+-----------------+-------------------+------------------+------------------------------------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
    |count  |891              |891                |891               |891                                             |891   |714               |891               |891                |891               |891              |204  |891     |
    |mean   |446.0            |0.3838383838383838 |2.308641975308642 |null                                            |null  |29.69911764705882 |0.5230078563411896|0.38159371492704824|260318.54916792738|32.2042079685746 |null |null    |
    |stddev |257.3538420152301|0.48659245426485753|0.8360712409770491|null                                            |null  |14.526497332334035|1.1027434322934315|0.8060572211299488 |471609.26868834975|49.69342859718089|null |null    |
    |min    |1                |0                  |1                 |"Andersson, Mr. August Edvard (""Wennerstrom"")"|female|0.42              |0                 |0                  |110152            |0.0              |A10  |C       |
    |max    |891              |1                  |3                 |van Melkebeke, Mr. Philemon                     |male  |80.0              |8                 |6                  |WE/P 5735         |512.3292         |T    |S       |
    +-------+-----------------+-------------------+------------------+------------------------------------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
    



```python
#From the above statistics we can clearly see that age and cabin columns have null values 
```


```python
#Checking for number of null values in age column
Train_DF.where(Train_DF['Age'].isNull()).count()
```




    177




```python
#Finding the average value of age column
Train_DF.describe('Age').show(truncate = False)
```

    +-------+------------------+
    |summary|Age               |
    +-------+------------------+
    |count  |714               |
    |mean   |29.69911764705882 |
    |stddev |14.526497332334035|
    |min    |0.42              |
    |max    |80.0              |
    +-------+------------------+
    



```python
#Checking for null values in Cabin column
Train_DF.where(Train_DF['Cabin'].isNull()).count()
```




    687




```python
#Replacing the null values in age column with average value of age column
Train_DF = Train_DF.fillna({'Age':29.69911764705882})
```


```python
#Again Checking for null values in age column
Train_DF.where(Train_DF['Age'].isNull()).count()
```




    0




```python
Train_DF.groupBy('Cabin').count().orderBy('count', ascending = False).show(Train_DF.count())
```

    +---------------+-----+
    |          Cabin|count|
    +---------------+-----+
    |           null|  687|
    |    C23 C25 C27|    4|
    |        B96 B98|    4|
    |             G6|    4|
    |           E101|    3|
    |        C22 C26|    3|
    |            F33|    3|
    |             F2|    3|
    |              D|    3|
    |            D17|    2|
    |           E121|    2|
    |           C123|    2|
    |            B20|    2|
    |            B28|    2|
    |            E25|    2|
    |            C65|    2|
    |            D20|    2|
    |            B22|    2|
    |            E67|    2|
    |           C126|    2|
    |            B18|    2|
    |            C52|    2|
    |          F G73|    2|
    |            C83|    2|
    |             F4|    2|
    |            B77|    2|
    |            E24|    2|
    |            D33|    2|
    |           C124|    2|
    |            C92|    2|
    |             E8|    2|
    |            D36|    2|
    |            C68|    2|
    |             C2|    2|
    |            D26|    2|
    |            D35|    2|
    |B57 B59 B63 B66|    2|
    |             B5|    2|
    |            E33|    2|
    |            B49|    2|
    |            B35|    2|
    |    B51 B53 B55|    2|
    |           C125|    2|
    |            E44|    2|
    |        B58 B60|    2|
    |            C93|    2|
    |            C78|    2|
    |            A23|    1|
    |            B79|    1|
    |           C110|    1|
    |             D7|    1|
    |            C95|    1|
    |            B39|    1|
    |            D21|    1|
    |             A6|    1|
    |            E31|    1|
    |           C128|    1|
    |            C90|    1|
    |            B30|    1|
    |            E50|    1|
    |           C104|    1|
    |            B50|    1|
    |              T|    1|
    |            A36|    1|
    |            D48|    1|
    |          F E69|    1|
    |            D28|    1|
    |           C103|    1|
    |            D15|    1|
    |            D45|    1|
    |        C62 C64|    1|
    |            B38|    1|
    |            E63|    1|
    |            C50|    1|
    |            C45|    1|
    |            E77|    1|
    |            B80|    1|
    |            A19|    1|
    |             B4|    1|
    |            E10|    1|
    |            C54|    1|
    |            C82|    1|
    |            D46|    1|
    |            E49|    1|
    |          F G63|    1|
    |            A32|    1|
    |            B71|    1|
    |            C87|    1|
    |            C86|    1|
    |            B86|    1|
    |             D9|    1|
    |            A20|    1|
    |            B94|    1|
    |        D10 D12|    1|
    |            D49|    1|
    |            D37|    1|
    |           B102|    1|
    |            A14|    1|
    |             A7|    1|
    |             C7|    1|
    |           C111|    1|
    |            E12|    1|
    |            C30|    1|
    |            A16|    1|
    |            B69|    1|
    |            B19|    1|
    |            A26|    1|
    |            C99|    1|
    |            E38|    1|
    |            C46|    1|
    |            C85|    1|
    |           C106|    1|
    |            D47|    1|
    |             B3|    1|
    |            E68|    1|
    |            A34|    1|
    |            E58|    1|
    |            E46|    1|
    |            E36|    1|
    |            A10|    1|
    |            D30|    1|
    |            A31|    1|
    |            B37|    1|
    |            E40|    1|
    |           C148|    1|
    |            C91|    1|
    |            D19|    1|
    |            A24|    1|
    |            E34|    1|
    |            D50|    1|
    |           B101|    1|
    |            C47|    1|
    |            E17|    1|
    |            B73|    1|
    |             A5|    1|
    |            C32|    1|
    |            F38|    1|
    |            D11|    1|
    |             D6|    1|
    |           C101|    1|
    |           C118|    1|
    |            B78|    1|
    |            C49|    1|
    |            C70|    1|
    |            D56|    1|
    |        B82 B84|    1|
    |            B42|    1|
    |            B41|    1|
    +---------------+-----+
    



```python
#Since there are more than 50% null values in cabin column we do not consider replacing the null values
```


```python
#To check frequency of data by Survival
Train_DF.groupBy('Survived').count().show()
```

    +--------+-----+
    |Survived|count|
    +--------+-----+
    |       1|  342|
    |       0|  549|
    +--------+-----+
    



```python
#To check frequency of data by type of Pclass
Train_DF.groupBy('Pclass').count().show()
```

    +------+-----+
    |Pclass|count|
    +------+-----+
    |     1|  216|
    |     3|  491|
    |     2|  184|
    +------+-----+
    



```python
#To check frequency of data by type of Gender
Train_DF.groupBy('Sex').count().show()
```

    +------+-----+
    |   Sex|count|
    +------+-----+
    |female|  314|
    |  male|  577|
    +------+-----+
    



```python
#To check frequency of data by age
Train_DF.groupBy('Embarked').count().show()
```

    +--------+-----+
    |Embarked|count|
    +--------+-----+
    |       Q|   78|
    |       C|  168|
    |       S|  645|
    +--------+-----+
    



```python
#To check average of data by number of people survival status 
Train_DF.groupBy('Survived').mean().show()
```

    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    |Survived|  avg(PassengerId)|avg(Survived)|       avg(Pclass)|          avg(Age)|         avg(SibSp)|        avg(Parch)|         avg(Fare)|
    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    |       1|444.36842105263156|          1.0|1.9502923976608186| 28.54977812177503|0.47368421052631576|0.4649122807017544| 48.39540760233917|
    |       0| 447.0163934426229|          0.0|2.5318761384335153|30.415099646415896| 0.5537340619307832|0.3296903460837887|22.117886885245877|
    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    



```python
#Question(1).What is the shape of the data contained in training.csv?
#=====================================================================
#(Answer):The shape of the data is the number of rows and columns (m = rows, n = columns) present in the dataset
#Seeing the shape of the dataset
print("Shape:", (Train_DF.count(), len(Train_DF.columns)))
```

    Shape: (891, 12)



```python
#Question(2).What features (or attributes) are recorded for each passenger in training.csv?
#(Answer):The features recorded for each automobile can be known by the column names in the dataframe
Train_DF_Features = Train_DF.columns
print("The features recorded for each automobile are :", Train_DF_Features)
```

    The features recorded for each automobile are : ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']



```python
#Question3:Provide a schema of the columns to be included in your model for this assignment. 
#Looking at the schema of the dataset
Train_DF.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = false)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
    



```python
# Schema for the model
model_schema = Train_DF.select('Survived','Pclass','Age','Sex','Embarked')
model_schema.printSchema()
```

    root
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Age: double (nullable = false)
     |-- Sex: string (nullable = true)
     |-- Embarked: string (nullable = true)
    



```python
#Question3.Provide a schema of the columns to be included in your model for this assignment. 
#Comment on columns that may require transformation(s). 
#An example of transformation is that of creating dummy variables. 
#List these columns and explain why and what transformation is required. I
#Include these comments in your notebook. 
#=======================================================================================
#(Answer): The columns that we would be using for this model are,
#Survived(Dependent Variable)
#INDEPENDENT VARIABLES
#Pclass(Categorical)
#Sex(Categorical)
#Age
#Embarked(Categorical)
#=======================================================================================
#In order to perform a logistic regression we should have independent variables as numerical datatype values
#So the columns to be transformed from categorical variable to numeric data are,
#-Pclass
#-Sex
#-Embarked
#The Pclass variable Pclass has three different categories 1,2,3 in order to use the column for logistic regression we need to create a Dummy Vector Variable using Encoder
#The categorical variable Geneder has either male or female labels for which we assign a indexer say 0 for male and 1 for female and use encoder to create a significant dummy vector varibles
#The embarked variable has three different labels Q,S,C for which first create string index and then assign respective numerical dummy vectors using encoder

```


```python
#Question(4):4.Comment on the balance of data in training.csv with regards to each input variable as well as your target variable. Support your comments with appropriate statistics.
Train_DF.groupBy('Survived').mean().show()
#people with an average of 28 are survived compared people above average of 30
```

    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    |Survived|  avg(PassengerId)|avg(Survived)|       avg(Pclass)|          avg(Age)|         avg(SibSp)|        avg(Parch)|         avg(Fare)|
    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    |       1|444.36842105263156|          1.0|1.9502923976608186| 28.54977812177503|0.47368421052631576|0.4649122807017544| 48.39540760233917|
    |       0| 447.0163934426229|          0.0|2.5318761384335153|30.415099646415896| 0.5537340619307832|0.3296903460837887|22.117886885245877|
    +--------+------------------+-------------+------------------+------------------+-------------------+------------------+------------------+
    



```python
# In case of balance in our original data 
#For Pclass we can see the data we have is biased towards a Pclass 3(which is third class passengers)
Train_DF.groupBy('Pclass').count().show()
```

    +------+-----+
    |Pclass|count|
    +------+-----+
    |     1|  216|
    |     3|  491|
    |     2|  184|
    +------+-----+
    



```python
# In case of balance in our original data 
#For Age we can see the data we have is balanced towards as it has people of all age categories
Train_DF.groupBy('Age').count().show()
```

    +----+-----+
    | Age|count|
    +----+-----+
    | 8.0|    4|
    |70.0|    2|
    | 7.0|    3|
    |20.5|    1|
    |49.0|    6|
    |29.0|   20|
    |40.5|    2|
    |64.0|    2|
    |47.0|    9|
    |42.0|   13|
    |24.5|    1|
    |44.0|    9|
    |35.0|   18|
    |62.0|    4|
    |18.0|   26|
    |80.0|    1|
    |34.5|    1|
    |39.0|   14|
    | 1.0|    7|
    |45.5|    2|
    +----+-----+
    only showing top 20 rows
    



```python
# In case of balance in our original data 
#For Sex we can see the data we have is biased towards Male because there are more male passengers data than female
Train_DF.groupBy('Sex').count().show()
```

    +------+-----+
    |   Sex|count|
    +------+-----+
    |female|  314|
    |  male|  577|
    +------+-----+
    



```python
# In case of balance in our original data 
#For Embarked we can see the data we have is biased towards s for Southampton, because there are more records with port of embarkation as S
Train_DF.groupBy('Embarked').count().show()
```

    +--------+-----+
    |Embarked|count|
    +--------+-----+
    |       Q|   78|
    |       C|  168|
    |       S|  645|
    +--------+-----+
    



```python
# In case of balance in our original data 
#For Survived we can see the data we have is biased towards people not survived, because there are more records Survived =0 compared to survival= 1
Train_DF.groupBy('Survived').count().show()
```

    +--------+-----+
    |Survived|count|
    +--------+-----+
    |       1|  342|
    |       0|  549|
    +--------+-----+
    



```python
#Question(4)a: Transformations
#a.Perform the transformations, if any, identified in step # 3. Perform feature engineering if and where needed, including Vectorization of relevant input variables. Provide a printout of the schema of your feature-engineered data.
#(Answer):
#Categorical values into numerical values automatically first for Pclass variables(Specification)
#StringIndexer arguments = name of input columns and resulting column
```


```python
#4(a)Categorical values into numerical values automatically first for Gender variables(Specification)
#StringIndexer arguments = name of input columns and resulting column
Sex_Indexer = StringIndexer(inputCol = 'Sex', outputCol = 'Gender_Num').fit(Train_DF)
```


```python
#4(a)Taking Categorical data and transforming
Train_DF = Sex_Indexer.transform(Train_DF)
```


```python
#4(a)Checking if numbers are assingned
Train_DF.show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Gender_Num|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|       0.0|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|       1.0|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|       1.0|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|       1.0|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|       0.0|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+
    only showing top 5 rows
    



```python
#4(a)To check count of data by type of gender after creating Gender_Num
Train_DF.groupBy('Gender_Num').count().show()
```

    +----------+-----+
    |Gender_Num|count|
    +----------+-----+
    |       0.0|  577|
    |       1.0|  314|
    +----------+-----+
    



```python
#4(a)Categorical values into numerical values automatically first for Pclass variables(Specification)
#StringIndexer arguments = name of input columns and resulting column
Embarked_Indexer = StringIndexer(inputCol = 'Embarked', outputCol = 'Embarked_Num').fit(Train_DF)
```


```python
#4(a)Taking Categorical data and transforming
Train_DF = Embarked_Indexer.transform(Train_DF)
```


```python
#4(a)Checking if numbers are assingned
Train_DF.show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Gender_Num|Embarked_Num|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|       0.0|         0.0|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|       1.0|         1.0|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|       1.0|         0.0|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|       1.0|         0.0|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|       0.0|         0.0|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+
    only showing top 5 rows
    



```python
#4(a)To check count of data by type of embarkment after creating Embarked_Num
Train_DF.groupBy('Embarked_Num').count().show()
```

    +------------+-----+
    |Embarked_Num|count|
    +------------+-----+
    |         0.0|  645|
    |         1.0|  168|
    |         2.0|   78|
    +------------+-----+
    



```python
#4(a)Check how many distinct values the variables have and assign right number of dummy variables for Pclass
Pclass_Encoder = OneHotEncoder(inputCol = 'Pclass', outputCol = 'Pclass_Dummy_Vector')
Train_DF = Pclass_Encoder.transform(Train_DF)
Train_DF.show(3)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Gender_Num|Embarked_Num|Pclass_Dummy_Vector|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|       0.0|         0.0|          (3,[],[])|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|       1.0|         1.0|      (3,[1],[1.0])|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|       1.0|         0.0|          (3,[],[])|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+
    only showing top 3 rows
    



```python
Train_DF.groupBy('Pclass_Dummy_Vector').count().show()
```

    +-------------------+-----+
    |Pclass_Dummy_Vector|count|
    +-------------------+-----+
    |      (3,[2],[1.0])|  184|
    |      (3,[1],[1.0])|  216|
    |          (3,[],[])|  491|
    +-------------------+-----+
    



```python
#4(a)Check how many distinct values the variables have and assign right number of dummy variables for Gender
Gender_Encoder = OneHotEncoder(inputCol = 'Gender_Num', outputCol = 'Gender_Dummy_Vector')
Train_DF = Gender_Encoder.transform(Train_DF)
Train_DF.show(3)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Gender_Num|Embarked_Num|Pclass_Dummy_Vector|Gender_Dummy_Vector|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|       0.0|         0.0|          (3,[],[])|      (1,[0],[1.0])|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|       1.0|         1.0|      (3,[1],[1.0])|          (1,[],[])|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|       1.0|         0.0|          (3,[],[])|          (1,[],[])|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+
    only showing top 3 rows
    



```python
Train_DF.groupBy('Gender_Dummy_Vector').count().show()
```

    +-------------------+-----+
    |Gender_Dummy_Vector|count|
    +-------------------+-----+
    |      (1,[0],[1.0])|  577|
    |          (1,[],[])|  314|
    +-------------------+-----+
    



```python
#4(a)Check how many distinct values the variables have and assign right number of dummy variables for Embarked
Embarked_Encoder = OneHotEncoder(inputCol = 'Embarked_Num', outputCol = 'Embarked_Dummy_Vector')
Train_DF = Embarked_Encoder.transform(Train_DF)
Train_DF.show(3)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+---------------------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Gender_Num|Embarked_Num|Pclass_Dummy_Vector|Gender_Dummy_Vector|Embarked_Dummy_Vector|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+---------------------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|       0.0|         0.0|          (3,[],[])|      (1,[0],[1.0])|        (2,[0],[1.0])|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|       1.0|         1.0|      (3,[1],[1.0])|          (1,[],[])|        (2,[1],[1.0])|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|       1.0|         0.0|          (3,[],[])|          (1,[],[])|        (2,[0],[1.0])|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+----------+------------+-------------------+-------------------+---------------------+
    only showing top 3 rows
    



```python
Train_DF.groupBy('Embarked_Dummy_Vector').count().show()
```

    +---------------------+-----+
    |Embarked_Dummy_Vector|count|
    +---------------------+-----+
    |        (2,[0],[1.0])|  645|
    |        (2,[1],[1.0])|  168|
    |            (2,[],[])|   78|
    +---------------------+-----+
    



```python
#Checking the Schema
Train_DF.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = false)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- Gender_Num: double (nullable = false)
     |-- Embarked_Num: double (nullable = false)
     |-- Pclass_Dummy_Vector: vector (nullable = true)
     |-- Gender_Dummy_Vector: vector (nullable = true)
     |-- Embarked_Dummy_Vector: vector (nullable = true)
    



```python
#4(a)Feature Engineering
#import transformer to combine all variables into one column to perform linear regression btw output and variables
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
```


```python
Assembler_Train_DF = VectorAssembler(inputCols = ['Age','Gender_Dummy_Vector','Pclass_Dummy_Vector','Embarked_Dummy_Vector'], outputCol = 'features')
```


```python
Train_DF = Assembler_Train_DF.transform(Train_DF)
```


```python
##4(a)printout of the schema of your feature-engineered data.
Train_DF.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = false)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- Gender_Num: double (nullable = false)
     |-- Embarked_Num: double (nullable = false)
     |-- Pclass_Dummy_Vector: vector (nullable = true)
     |-- Gender_Dummy_Vector: vector (nullable = true)
     |-- Embarked_Dummy_Vector: vector (nullable = true)
     |-- features: vector (nullable = true)
    



```python
#Dataset for generation of model
Model_DF = Train_DF.select(['Survived','features'])
```


```python
#Schema of the model dataset
Model_DF.printSchema()
```

    root
     |-- Survived: integer (nullable = true)
     |-- features: vector (nullable = true)
    



```python
Model_DF.show(3, truncate = False)
```

    +--------+--------------------------+
    |Survived|features                  |
    +--------+--------------------------+
    |0       |(7,[0,1,5],[22.0,1.0,1.0])|
    |1       |(7,[0,3,6],[38.0,1.0,1.0])|
    |1       |(7,[0,5],[26.0,1.0])      |
    +--------+--------------------------+
    only showing top 3 rows
    



```python
#Question(4)b.To train and then test your model, split the data from training.csv into training and test datasets using an 80/20 split. 
#Like you did in step 4 above, comment on the balance of data in the training and test datasets. 
#Are they representative of the overall data? 
#What can you say about the balance in target classes in both the training and test datasets?
#Answer)
#Splitting of data for regression into required division
Training_DF, Testing_DF = Model_DF.randomSplit([0.75,0.25])
```


```python
print(Training_DF.count())
```

    678



```python
print(Testing_DF.count())
```

    213



```python
#In training dataset there are about 270 survived passengers and 449 of unsurvived passengers
Training_DF.groupBy('Survived').count().show()
```

    +--------+-----+
    |Survived|count|
    +--------+-----+
    |       1|  272|
    |       0|  406|
    +--------+-----+
    



```python
#The testing dataset has about 72 survived passengers and 100 passengers who are not survived
Testing_DF.groupBy('Survived').count().show()
```

    +--------+-----+
    |Survived|count|
    +--------+-----+
    |       1|   70|
    |       0|  143|
    +--------+-----+
    



```python
#This ensures we have a balance set of the target class (Survived) into the training and test set.
```


```python
#Question(4)c: Build and train the Logistic Regression model. 
#Generate a list of predictions for passengers survival status (survival = 1) based on the trained model. 
#Display actual, predicted, and probability values for the first 10 rows only. Based on these results, comment on the performance of the model?
#Is the model predicting likelihood of survival with high probability? 
#Using Logistic Regression
Log_Reg = LogisticRegression(labelCol = 'Survived').fit(Training_DF)
```


```python
#Generating the predictions
Training_Results = Log_Reg.evaluate(Training_DF).predictions
```


```python
#Filtering predictions and trained results which are equal to 1
#Applied a non-linear function to generate probability
#So, in the above results, probability at the 0th index is for Status = 0 and probability as 1st index is for Status =1.
Training_Results.filter(Training_Results['Survived']==1).filter(Training_Results['prediction']==1).select(['Survived','prediction','probability']).show(10, truncate = False)
```

    +--------+----------+----------------------------------------+
    |Survived|prediction|probability                             |
    +--------+----------+----------------------------------------+
    |1       |1.0       |[0.21945587715809511,0.7805441228419049]|
    |1       |1.0       |[0.2243665398869764,0.7756334601130236] |
    |1       |1.0       |[0.2243665398869764,0.7756334601130236] |
    |1       |1.0       |[0.23956318504334329,0.7604368149566567]|
    |1       |1.0       |[0.2554501292095873,0.7445498707904127] |
    |1       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |1       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |1       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |1       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |1       |1.0       |[0.2992689360778693,0.7007310639221306] |
    +--------+----------+----------------------------------------+
    only showing top 10 rows
    



```python
#Answer: in the above 10 rows we can see that the actual survived value ansd predicted survival value are same for the top 10 rows.
#So the second probabibility values are likely hood of survival which is predicted with high probability by the model
```


```python
#Question:4(d)
#d.	Using the test data from the 80/20 split, evaluate the performance of your trained model.
#Compute and show the values for Accuracy, Recall, Precision, and an F1 score. 
#Comment of general usefulness of the model in predicting the survival status of passengers given their age, gender, pclass and embarked values. 
#Answer:
#Generating the predicted values using test data
#We can see that the model is pretty good and useful as its generating output with a high probability similar to actual data
Results = Log_Reg.evaluate(Testing_DF).predictions
```


```python
#Applied a non-linear function to generate probability for test dataset
Results.select('Survived','prediction','probability').show(5, truncate = False)
```

    +--------+----------+----------------------------------------+
    |Survived|prediction|probability                             |
    +--------+----------+----------------------------------------+
    |0       |1.0       |[0.23442044032462525,0.7655795596753747]|
    |0       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |0       |1.0       |[0.2992689360778693,0.7007310639221306] |
    |0       |0.0       |[0.8017663671050823,0.19823363289491774]|
    |0       |0.0       |[0.8275007792065519,0.17249922079344812]|
    +--------+----------+----------------------------------------+
    only showing top 5 rows
    



```python
#Since this is a classification problem, we will use a confusion matrix togauge the performance of the model.
#Seeing the predicted values for test data which shows TP(True Positives),FP(False Positives),TN(True Negatives),FN(False Negatives)
```


```python
#For number of True Positives
TP = Results[(Results.Survived == 1) & (Results.prediction == 1)].count()
```


```python
#For number of True Negatives
TN = Results[(Results.Survived == 0) & (Results.prediction == 0)].count()
```


```python
#For number of False Positives(Actual=0,Predicted =1)
FP = Results[(Results.Survived == 0) & (Results.prediction == 1)].count()
```


```python
#For number of False Negatives(Actual=1,Predicted =0)
FN = Results[(Results.Survived == 1) & (Results.prediction == 0)].count()
```


```python
#Manual computations
Accuracy = (TP + TN)/(TP+TN+FP+FN)
Accuracy
```




    0.7981220657276995




```python
#Recall rate shows how much of the positive class cases we are able topredict correctly out of the total positive class observations.
Recall = TP/(TP + FN)
Recall
```




    0.7285714285714285




```python
#Precision rate talks about the number of true positives predicted correctly out of all the predicted positives observations:
Precision = TP/(TP + FP)
Precision
```




    0.68




```python
#summary of the model trained with training dataset
Training_Summary = Log_Reg.summary
```


```python
#Calculating ROC
print("areaUnderROC: " + str(Training_Summary.areaUnderROC))
```

    areaUnderROC: 0.8498533033903216



```python
print(Training_Summary.accuracy)
print(Training_Summary.weightedRecall)
print(Training_Summary.weightedPrecision)
print(Training_Summary.weightedFMeasure())

```

    0.7876106194690266
    0.7876106194690264
    0.7864204728666768
    0.7867617751311864



```python
#So, the recall rate and precision rate are also in the same range in the above calculations, which is shows target class was well balanced.
# We can say that the model performance is good as  Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s
```
