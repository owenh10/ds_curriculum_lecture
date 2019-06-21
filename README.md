
# Jupyter Notebooks and Data Visualization

A jupyter notebook is a document that is cabale of displaying and executing python code alongside plain text. This document--and the document used earlier in the Data Science curriculum--are both examples of jupyter notebooks. The notebook is held in an `.ipynb` file (a 'notebook'), and is opened from a locally hosted server in your browser. To get our environment set up, let's first start a virtual environment in our terminal:

```
$ python -m venv visualization
$ source visualization/bin/activate
```

Then, let's start up a jupyter server by running `jupyter notebook` in our terminal. This will open the jupyter GUI in your browser. There, let's hit `new` and select Python3 from the dropown menu. We now have a jupyter notebook with a Python kernel--you can look in the folder you ran `jupyter notebook` in and observe that a new `.ipynb` file has been created there.

Now we need to install the necessary libraries. Create a new code cell in your jupyter notebook and run the following:


```python
import pandas as pd
import seaborn as sb
```

From now on, we will be running all our code in new code cells a bit at a time; the main benefit of jupyter is that we don't need to run our whole file at once, so let's take advantage of this!

### Reading in Data
Now we want to find some data to read in--our csv of iris data should be good. We can use pandas' read_csv method to put our data into a dataframe:


```python
iris = pd.read_csv("Iris.csv")
```

Let's make sure out data is loaded correctly by calling pandas' "head" method, which displays the first five rows of a dataframe:


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



One thing we might want to do with other data sets, potentially to clarify what variable is being observed or maybe even just because we like another name more. We can use the rename function in Pandas to do this. If we want Id to read as IrisId, we simply run:


```python
iris.rename(columns={'Id':'IrisId'}, inplace=True)
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IrisId</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



Easy! To clarify, we used `inplace=True` to specify that our change is being made on the original dataframe and is not being saved as a new object.
### Simple Description
Let's think of our work today as being motivated by a simple mission; we want to be able to identify interesting or unusual trends in our data, trends which might allow us to build predictive tools based on our data. One thing we might want to be able to with this data is predict the species of an iris based on it petal length, petal width, sepal length, and sepal width. Let's start, however, with some basic summary statistics.

We can get summary statistics of the characters of our irises using pandas's `describe()` method.


```python
iris.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IrisId</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75.500000</td>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43.445368</td>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38.250000</td>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75.500000</td>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.750000</td>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150.000000</td>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



We can do this for just one of our species by selecting only observations where the 'Species' value matches the one we want. For example,


```python
iris[iris.Species=='Iris-setosa'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IrisId</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.00000</td>
      <td>50.00000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.50000</td>
      <td>5.00600</td>
      <td>3.418000</td>
      <td>1.464000</td>
      <td>0.24400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.57738</td>
      <td>0.35249</td>
      <td>0.381024</td>
      <td>0.173511</td>
      <td>0.10721</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>4.30000</td>
      <td>2.300000</td>
      <td>1.000000</td>
      <td>0.10000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.25000</td>
      <td>4.80000</td>
      <td>3.125000</td>
      <td>1.400000</td>
      <td>0.20000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25.50000</td>
      <td>5.00000</td>
      <td>3.400000</td>
      <td>1.500000</td>
      <td>0.20000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.75000</td>
      <td>5.20000</td>
      <td>3.675000</td>
      <td>1.575000</td>
      <td>0.30000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>50.00000</td>
      <td>5.80000</td>
      <td>4.400000</td>
      <td>1.900000</td>
      <td>0.60000</td>
    </tr>
  </tbody>
</table>
</div>



It looks like we forgot that the leftmost column is reserved for ids for each individual iris--not very interesting data considering it's just range(1,150). Let's create a more suitable dataframe by dropping the id column and then look at that dataframe's `description()`.


```python
irisCleaned = iris.drop('IrisId', axis=1)
irisCleaned[irisCleaned.Species=='Iris-setosa'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.00000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.00600</td>
      <td>3.418000</td>
      <td>1.464000</td>
      <td>0.24400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.35249</td>
      <td>0.381024</td>
      <td>0.173511</td>
      <td>0.10721</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.30000</td>
      <td>2.300000</td>
      <td>1.000000</td>
      <td>0.10000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.80000</td>
      <td>3.125000</td>
      <td>1.400000</td>
      <td>0.20000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.00000</td>
      <td>3.400000</td>
      <td>1.500000</td>
      <td>0.20000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.20000</td>
      <td>3.675000</td>
      <td>1.575000</td>
      <td>0.30000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.80000</td>
      <td>4.400000</td>
      <td>1.900000</td>
      <td>0.60000</td>
    </tr>
  </tbody>
</table>
</div>



Here we have summary statstics for each variable; if we wanted to show summary statistics by variable for each individual species, it would be a simple matter of grouping by column "Species" and running `describe()` once more.


```python
irisCleaned.groupby("Species").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">SepalLengthCm</th>
      <th colspan="2" halign="left">SepalWidthCm</th>
      <th>...</th>
      <th colspan="2" halign="left">PetalLengthCm</th>
      <th colspan="8" halign="left">PetalWidthCm</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iris-setosa</th>
      <td>50.0</td>
      <td>5.006</td>
      <td>0.352490</td>
      <td>4.3</td>
      <td>4.800</td>
      <td>5.0</td>
      <td>5.2</td>
      <td>5.8</td>
      <td>50.0</td>
      <td>3.418</td>
      <td>...</td>
      <td>1.575</td>
      <td>1.9</td>
      <td>50.0</td>
      <td>0.244</td>
      <td>0.107210</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>Iris-versicolor</th>
      <td>50.0</td>
      <td>5.936</td>
      <td>0.516171</td>
      <td>4.9</td>
      <td>5.600</td>
      <td>5.9</td>
      <td>6.3</td>
      <td>7.0</td>
      <td>50.0</td>
      <td>2.770</td>
      <td>...</td>
      <td>4.600</td>
      <td>5.1</td>
      <td>50.0</td>
      <td>1.326</td>
      <td>0.197753</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>1.5</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>Iris-virginica</th>
      <td>50.0</td>
      <td>6.588</td>
      <td>0.635880</td>
      <td>4.9</td>
      <td>6.225</td>
      <td>6.5</td>
      <td>6.9</td>
      <td>7.9</td>
      <td>50.0</td>
      <td>2.974</td>
      <td>...</td>
      <td>5.875</td>
      <td>6.9</td>
      <td>50.0</td>
      <td>2.026</td>
      <td>0.274650</td>
      <td>1.4</td>
      <td>1.8</td>
      <td>2.0</td>
      <td>2.3</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>



We unfortunately can't see all our data; if we wanted to get summary statstics for every variable for every Species all on the same page, we would most likely have to do each variable seperately. In any case, it still wouldn't tell us much--numbers don't give us a great idea of the real differences between observations, at least not at first glance. If we want to look at each individual point as part of the whole, what we want is a nice visualization for our data that gives us information at a glance.

### Simple Visualization

So far, all we've done is dataframe manipulation in pands. This looks fine, but what we really want to do is present our data visually and in an easy-to understand way. We'd much rather look at a graph than a bunch of numbers, so our next step is to produce a scatterplot of our data. Remember, a scatterplot labels the x and y axes with aspects of our data and then puts a dot where each individual iris falls. For our first comparison, we want to see Sepal Length on the x-axis and Sepal Width on the y-axis. A relplot will allow us to show the relationship between two categories of quantitative data.


```python
sb.relplot(data=iris, x='SepalLengthCm', y='SepalWidthCm')
```




    <seaborn.axisgrid.FacetGrid at 0x11c11a630>




![png](output_23_1.png)


While this looks very convincing, it doesn't tell us much about the data we're dealing with. In fact, we are unable to infer anything about our data except that all our sepal lengths are between 4.5 and 8 centimeters, and all our sepal widths are between 2 and 4.5 centimeters. Let's add another aspect to our visualization by showing the species of each of our individual irises on our plot by coloring each point based on its species.

### Now, with color!


```python
sb.relplot(data=iris, kind='scatter', x='SepalLengthCm', y='SepalWidthCm', hue='Species')
```




    <seaborn.axisgrid.FacetGrid at 0x1048a07f0>




![png](output_26_1.png)




Seaborn makes coloring points by species very easy; all we have to specify is what column our color is going to be based on, and it takes care of the rest. Unfortunately, we have found a little bit of an issue with our visualization by coloring it--although it's pretty easy to tell whether an iris is a setosa based on this graph, there's a lot of overlap between versicolors and virginicas. We'll get into how we can discover this through machine learning later (!), but for right now let's have a look at some other graphical representations of our data.

### Box Plot


```python
sb.boxplot(data=(iris.drop('IrisId', axis=1)))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x104987ef0>




![png](output_30_1.png)




This looks better, but it's still not perfect. Now we can see the distribution for the whole data set, but we can't see which species each point belongs to. This is what's called a wide-form data format--each row has many variables (columns) and we can plot them all on the same graph. Unfortunately, Seaborn doesn't allow us to color by species on a wide-form plot (go ahead and try running `plot = sb.swarmplot(data=(iris.drop('Id', axis=1)), hue="Species")`--you'll get a `No hue grouping with wide inputs` error). That's okay--we have an even better way to look at our data and determine which variable--or combination of variables--will be the most helpful in classifying iris species based on field measurements.

### Pair Plot


```python
sb.pairplot(data=iris.drop('IrisId', axis=1), hue='Species')
```




    <seaborn.axisgrid.PairGrid at 0x11c5d1e48>




![png](output_34_1.png)


This graph is an example of both a correllation matrix and a pair plot. A correlation matrix shows us the relationships between variables graphically, and a pair plot is just a scatterplot where both axes are variables in our data. This graph lets us quickly compare the utility of variable combinations in trying to predict the species of a given iris based only on our 4 observed variables. Notice how easy it was to generate with Seaborn!

## Assignments

* [Texas Senate Race](https://github.com/indiaplatoon/senate-race)
