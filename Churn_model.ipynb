{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Churn Prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The project we're describing involves churn prediction, which is a common application in the field of machine learning. **Churn refers to the phenomenon where customers decide to leave a company.** Predicting churn is crucial for businesses as **it helps them understand why and when customers are likely to stop using their services.** By building an accurate churn prediction model, companies can take proactive measures to retain customers and prevent them from leaving.\n",
    "\n",
    "Here's a breakdown of the project's key details:\n",
    "\n",
    "**Dataset Information:**\n",
    "- The dataset we're using is titled \"Telco Customer Churn,\" and it's available on Kaggle.\n",
    "- It comprises information on 7043 customers.\n",
    "- The dataset includes 20 independent variables (features) and 1 dependent variable (target).\n",
    "- The target variable indicates whether a customer has left the company recently (churn=yes) or not.\n",
    "####  Since the target variable has two states (yes/no or 1/0), this is a binary classification problem.\n",
    "\n",
    "**Features (Independent Variables):**\n",
    "1. `customerID`: Unique identifier for each customer (irrelevant for churn prediction).\n",
    "2. `gender`: Gender of the customer.\n",
    "3. `SeniorCitizen`: Whether the customer is a senior citizen or not (1 for yes, 0 for no).\n",
    "4. `Partner`: Whether the customer has a partner or not (Yes or No).\n",
    "5. `Dependents`: Whether the customer has dependents or not (Yes or No).\n",
    "6. `tenure`: Number of months the customer has stayed with the company.\n",
    "7. `PhoneService`: Whether the customer has a phone service or not (Yes or No).\n",
    "8. `MultipleLines`: Whether the customer has multiple phone lines (Yes, No, No phone service).\n",
    "9. `InternetService`: Customer's internet service provider (DSL, Fiber optic, No internet service).\n",
    "10. `OnlineSecurity`: Whether the customer has online security (Yes, No, No internet service).\n",
    "11. `OnlineBackup`: Whether the customer has online backup (Yes, No, No internet service).\n",
    "12. `DeviceProtection`: Whether the customer has device protection (Yes, No, No internet service).\n",
    "13. `TechSupport`: Whether the customer has tech support (Yes, No, No internet service).\n",
    "14. `StreamingTV`: Whether the customer has streaming TV (Yes, No, No internet service).\n",
    "15. `StreamingMovies`: Whether the customer has streaming movies (Yes, No, No internet service).\n",
    "16. `Contract`: The contract term of the customer (Month-to-month, One year, Two years).\n",
    "17. `PaperlessBilling`: Whether the customer has paperless billing (Yes or No).\n",
    "18. `PaymentMethod`: The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card).\n",
    "19. `MonthlyCharges`: The monthly amount charged to the customer.\n",
    "20. `TotalCharges`: The total amount charged to the customer over time.\n",
    "\n",
    "**Project Goals and Approach:**\n",
    "- The goal of the project is to build a predictive model that accurately forecasts whether a customer will churn or not based on the provided features.\n",
    "- The provided dataset seems comprehensive and contains potentially relevant information for churn prediction.\n",
    "- During the project, we'll likely perform \n",
    "       * data preprocessing, \n",
    "       * exploratory data analysis, \n",
    "       * feature selection or engineering, \n",
    "       * model selection, training, and evaluation.\n",
    "- By analyzing the relationships between features and the target variable, we aim to uncover insights that could help identify patterns associated with customer churn.\n",
    "- The ultimate aim is to deploy a robust churn prediction model that aids businesses in making informed decisions to retain customers effectively.\n",
    "\n",
    "In summary, this project involves using a dataset containing various customer attributes to predict whether customers are likely to churn from a company's services. Through thorough analysis and machine learning techniques, we aim to build an accurate model that can assist businesses in preventing customer attrition."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing the data set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the necessary libraries for data analysis and visualization\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Enable inline plotting in Jupyter notebooks\n",
    "%matplotlib inline\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read a CSV file into a Pandas DataFrame\n",
    "df = pd.read_csv(\"Telco-Customer-Churn.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>customerID</th>\n",
       "      <th>gender</th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>PhoneService</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>...</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>Contract</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>TotalCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7590-VHVEG</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>29.85</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5575-GNVDE</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>1889.5</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3668-QPYBK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>108.15</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7795-CFOCW</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>1840.75</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9237-HQITU</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>151.65</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \\\n",
       "0  7590-VHVEG  Female              0     Yes         No       1           No   \n",
       "1  5575-GNVDE    Male              0      No         No      34          Yes   \n",
       "2  3668-QPYBK    Male              0      No         No       2          Yes   \n",
       "3  7795-CFOCW    Male              0      No         No      45           No   \n",
       "4  9237-HQITU  Female              0      No         No       2          Yes   \n",
       "\n",
       "      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \\\n",
       "0  No phone service             DSL             No  ...               No   \n",
       "1                No             DSL            Yes  ...              Yes   \n",
       "2                No             DSL            Yes  ...               No   \n",
       "3  No phone service             DSL            Yes  ...              Yes   \n",
       "4                No     Fiber optic             No  ...               No   \n",
       "\n",
       "  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \\\n",
       "0          No          No              No  Month-to-month              Yes   \n",
       "1          No          No              No        One year               No   \n",
       "2          No          No              No  Month-to-month              Yes   \n",
       "3         Yes          No              No        One year               No   \n",
       "4          No          No              No  Month-to-month              Yes   \n",
       "\n",
       "               PaymentMethod MonthlyCharges  TotalCharges Churn  \n",
       "0           Electronic check          29.85         29.85    No  \n",
       "1               Mailed check          56.95        1889.5    No  \n",
       "2               Mailed check          53.85        108.15   Yes  \n",
       "3  Bank transfer (automatic)          42.30       1840.75    No  \n",
       "4           Electronic check          70.70        151.65   Yes  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7043, 21)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#missing values in the data set\n",
    "df.isna().sum().sum()    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is no missing value in the data set so we can jump to explore it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',\n",
       "       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',\n",
       "       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',\n",
       "       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',\n",
       "       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Retrieve and display the column names of the DataFrame\n",
    "\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "customerID           object\n",
       "gender               object\n",
       "SeniorCitizen         int64\n",
       "Partner              object\n",
       "Dependents           object\n",
       "tenure                int64\n",
       "PhoneService         object\n",
       "MultipleLines        object\n",
       "InternetService      object\n",
       "OnlineSecurity       object\n",
       "OnlineBackup         object\n",
       "DeviceProtection     object\n",
       "TechSupport          object\n",
       "StreamingTV          object\n",
       "StreamingMovies      object\n",
       "Contract             object\n",
       "PaperlessBilling     object\n",
       "PaymentMethod        object\n",
       "MonthlyCharges      float64\n",
       "TotalCharges         object\n",
       "Churn                object\n",
       "dtype: object"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display the data types of each column in the DataFrame\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Churn\n",
       "No     5174\n",
       "Yes    1869\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Churn.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Target variable has imbalanced class distribution. \n",
    "* Negative class (Churn=No) is much less than positive class (churn=Yes). \n",
    "* Imbalanced class distributions influence the performance of a machine learning model negatively. \n",
    "\n",
    "#### We will use upsampling or downsampling to overcome this issue. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is always beneficial to explore the features (independent variables) before trying to build a model. Let's first discover the features that only have two values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Retrieve column names with binary (two unique values) data\n",
    "columns = df.columns\n",
    "binary_cols = []\n",
    "\n",
    "for col in columns:\n",
    "    if df[col].value_counts().shape[0] == 2:\n",
    "        binary_cols.append(col)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['gender',\n",
       " 'SeniorCitizen',\n",
       " 'Partner',\n",
       " 'Dependents',\n",
       " 'PhoneService',\n",
       " 'PaperlessBilling',\n",
       " 'Churn']"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# categorical features with two classes\n",
    "binary_cols "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The remaining categorical variables have more than two values (or classes)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Categorical features with multiple classes\n",
    "multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',\n",
    " 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Binary categorical features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's check the class distribution of binary features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='PaperlessBilling', ylabel='count'>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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hx9+cHVpsIji19Czzz57yVhMTIxsNpueeuqpG18YUIHNnj1bDRo0kJeXl1q3bq1du3a5uiRUEJs3b9YjjzyioKAg2Ww2rVy50tUlAWXCMAxFRkYqKirqkrF3331Xfn5++ve//+2CylARFb+/fOONN5zWr1y5UjabzUVVWR/BvpLr3LmzTp486bSEhIS4uixYSHBwsJYuXaqzZ8+a686dO6clS5aoXr16LqwMqHg++ugjxcbGavz48frmm2/UrFkzRUVFKTs729WloQI4c+aMmjVrptmzZ7u6FKBM2Ww2LVy4UDt37tTcuXPN9enp6Ro9erRmzpypunXrurBCVDReXl568803dfr0aVeXUmkQ7Cs5T09PBQQEOC3u7u767LPP1LJlS3l5een222/XK6+8ogsXLpjb2Ww2zZ07Vw8//LCqVaum0NBQJScn64cfftCDDz4oHx8f/elPf9LRo0fNbY4eParu3bvL399f1atX17333quvvvrqsvXl5ORo8ODBql27tux2uzp06KB9+/aV288D165ly5YKDg7W8uXLzXXLly9XvXr11KJFC3Pd2rVrdf/998vPz0+1atXSww8/7PT6KMm3336rLl26qHr16vL391f//v313//+t9yOBShvU6dO1ZAhQ/T0008rLCxMc+bMUbVq1bRgwQJXl4YKoEuXLnr11Vf16KOPuroUoMwFBwdr+vTpGjlypNLT02UYhqKjo9WpUye1aNHisr/vP/nkE4WHh8vb21u1atVSZGSkzpw548KjQXmLjIxUQECAJk2a9IdzPv30U911113y9PRUgwYNNGXKlBtYofUQ7G9CW7Zs0YABA/TCCy/o0KFDmjt3rhISEvTaa685zZs4caIGDBig1NRUNWnSRE888YT++te/auzYsdqzZ48Mw9CwYcPM+Xl5eeratauSkpK0d+9ede7cWY888ogyMjL+sJbevXsrOztbX375pVJSUtSyZUt17NhRp06dKrfjx7UbNGiQFi5caD5esGCBnn76aac5Z86cUWxsrPbs2aOkpCS5ubnp0UcfVVFRUYn7zMnJUYcOHdSiRQvt2bNHa9euVVZWlh577LFyPRagvBQUFCglJUWRkZHmOjc3N0VGRio5OdmFlQHAjTFw4EB17NhRgwYN0qxZs/Ttt99q7ty5l/19f/LkSfXt21eDBg3S4cOHtXHjRvXs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AAAA16eKqwu4Ef773/+qsLBQ/v7+Tuv9/f313XffXTI/Pz9f+fn55uPc3FxJksPhuObnLsw/e83bwLVK8//z9fjlXOENfT5cv9K8Roq3MQyjrMu54cqqR9IfrelG9kj6o/XQH3kPeTOjP+JKrvU1ck390bgJ/Oc//zEkGdu3b3daP2rUKOO+++67ZP748eMNSSwsLCxlvhw/fvxGtb5yQ49kYWEpj4X+yMLCwlLycjX90WYYleDPo1dQUFCgatWq6ZNPPlGPHj3M9QMHDlROTo4+++wzp/m//2trUVGRTp06pVq1aslms92osis0h8Oh4OBgHT9+XHa73dXloALiNeLMMAz98ssvCgoKkpubtT8FRY+8PF77uBJeI87ojzcPXvu4El4jzq6lP94Ul+J7eHioVatWSkpKMoN9UVGRkpKSNGzYsEvme3p6ytPT02mdn5/fDajUeux2O//R4bJ4jfwfX19fV5dQJuiRV4fXPq6E18j/oT/eXHjt40p4jfyfq+2PN0Wwl6TY2FgNHDhQ99xzj+677z698847OnPmjJ5++mlXlwYAAAAAQKndNMH+8ccf108//aS4uDhlZmaqefPmWrt27SU31AMAAAAAwEpummAvScOGDSvx0ntcO09PT40fP/6Sy82AYrxGcLPitY8r4TWCmxWvfVwJr5HSuylungcAAAAAQGVl7VuPAgAAAABwkyPYAwAAAABgYQR7AAAAAAAsjGAPAAAAAICFEezxh5566inZbDa98cYbTutXrlwpm83moqrgSoZhKDIyUlFRUZeMvfvuu/Lz89O///1vF1QG3Fj0R5SEHombFa99XAt+h5YPgj0uy8vLS2+++aZOnz7t6lJQAdhsNi1cuFA7d+7U3LlzzfXp6ekaPXq0Zs6cqbp167qwQuDGoT/i9+iRuFnx2se14ndo2SPY47IiIyMVEBCgSZMm/eGcTz/9VHfddZc8PT3VoEEDTZky5QZWiBstODhY06dP18iRI5Weni7DMBQdHa1OnTqpRYsW6tKli6pXry5/f3/1799f//3vf81tP/nkE4WHh8vb21u1atVSZGSkzpw548KjAUqP/oiS0CNxs+K1j2vB79CyR7DHZbm7u+v111/XzJkzS7yEKiUlRY899pj69OmjAwcOaMKECXr55ZeVkJBw44vFDTNw4EB17NhRgwYN0qxZs/Ttt99q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",
      "text/plain": [
       "<Figure size 1200x700 with 6 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)\n",
    "\n",
    "sns.countplot(x = \"gender\", data=df, ax=axes[0,0])\n",
    "sns.countplot(x =\"SeniorCitizen\", data=df, ax=axes[0,1])\n",
    "sns.countplot(x =\"Partner\", data=df, ax=axes[0,2])\n",
    "sns.countplot(x =\"Dependents\", data=df, ax=axes[1,0])\n",
    "sns.countplot(x =\"PhoneService\", data=df, ax=axes[1,1])\n",
    "sns.countplot(x =\"PaperlessBilling\", data=df, ax=axes[1,2])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is a **high imbalance** in **SeniorCitizen** and **PhoneService** variables. Most of the customers are not senior and similarly, most customers have a phone service."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is better to check **how the target variable (churn) changes according to the binary features.** \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     No\n",
       "1     No\n",
       "2    Yes\n",
       "3     No\n",
       "4    Yes\n",
       "Name: Churn, dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Churn'].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**To be able to make calculations, we need to change the values of target variable. \"Yes\" will be 1 and \"No\" will be 0.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "churn_numeric = {'Yes':1, 'No':0}\n",
    "df.Churn.replace(churn_numeric, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Female</th>\n",
       "      <td>0.269209</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Male</th>\n",
       "      <td>0.261603</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Churn\n",
       "gender          \n",
       "Female  0.269209\n",
       "Male    0.261603"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Group by 'gender' and calculate the mean of 'Churn'\n",
    "\n",
    "df[['gender','Churn']].groupby(['gender']).mean()\n",
    "\n",
    "# This code helps us to understand how to churn rate(average value of 'churn')\n",
    "# Differs between genders"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Average churn rate** for **males** and **females** are approximately the same which **indicates gender variable does not bring a valuable prediction power to a model**. Therefore, I will not use gender variable in the machine learning model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.236062</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.416813</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Churn\n",
       "SeniorCitizen          \n",
       "0              0.236062\n",
       "1              0.416813"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['SeniorCitizen','Churn']].groupby(['SeniorCitizen']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Partner</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.329580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.196649</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Churn\n",
       "Partner          \n",
       "No       0.329580\n",
       "Yes      0.196649"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Partner','Churn']].groupby(['Partner']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Dependents</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.312791</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.154502</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Churn\n",
       "Dependents          \n",
       "No          0.312791\n",
       "Yes         0.154502"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Dependents','Churn']].groupby(['Dependents']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PhoneService</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.249267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.267096</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Churn\n",
       "PhoneService          \n",
       "No            0.249267\n",
       "Yes           0.267096"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['PhoneService','Churn']].groupby(['PhoneService']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.163301</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.335651</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Churn\n",
       "PaperlessBilling          \n",
       "No                0.163301\n",
       "Yes               0.335651"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**The other binary features have an effect on the target variable.** \n",
    "\n",
    "The phone service may also be skipped if you think 2% difference can be ignored. I have decided to use this feature in the model.\n",
    "\n",
    "You can also use pandas pivot_table function to check the relationship between features and target variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Female</th>\n",
       "      <td>0.239384</td>\n",
       "      <td>0.422535</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Male</th>\n",
       "      <td>0.232808</td>\n",
       "      <td>0.411150</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "SeniorCitizen         0         1\n",
       "gender                           \n",
       "Female         0.239384  0.422535\n",
       "Male           0.232808  0.411150"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "table = pd.pivot_table(df, values='Churn', index=['gender'],\n",
    "                    columns=['SeniorCitizen'], aggfunc=np.mean)\n",
    "table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Dependents</th>\n",
       "      <th>No</th>\n",
       "      <th>Yes</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Partner</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.342378</td>\n",
       "      <td>0.213296</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.254083</td>\n",
       "      <td>0.142367</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Dependents        No       Yes\n",
       "Partner                       \n",
       "No          0.342378  0.213296\n",
       "Yes         0.254083  0.142367"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "table = pd.pivot_table(df, values='Churn', index=['Partner'],\n",
    "                    columns=['Dependents'], aggfunc=np.mean)\n",
    "table"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Other Categorical Features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is time to explore other categorical features. We also have continuous features such as **tenure**, **monthly charges** and **total charges** which I will discuss in the next part.\n",
    "\n",
    "There are 6 variables that come with internet service. There variables come into play if customer has internet service."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "InternetService\n",
       "Fiber optic    3096\n",
       "DSL            2421\n",
       "No             1526\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['InternetService'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Internet Service"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='InternetService', ylabel='count'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AACA8QhEAADAeAENRDk5OerRo4eaNWum2NhYDR8+XDt37vSr6du3rxwOh98yceJEv5q9e/dqyJAhatKkiWJjYzVt2jSdOHHCr2bjxo3q1q2bnE6n2rRpo9zc3HN9eQAA4AIR0EC0adMmZWRk6JNPPlF+fr6OHz+ugQMH6ujRo35148eP14EDB+xl3rx59lhVVZWGDBmiyspKffzxx1q6dKlyc3M1Y8YMu2bPnj0aMmSI+vXrp6KiIk2ZMkV333231q5de96uFQAABK/GgTz5mjVr/NZzc3MVGxurwsJC9enTx97epEkTud3uUx5j3bp1+vrrr/X+++8rLi5OXbt21dy5c5WVlaVZs2YpLCxMixYtUnJysp544glJUocOHfTRRx9p/vz5SktLq3XMiooKVVRU2Os+n68hLhcAAASpoJpDVF5eLkmKjo72275s2TJdcskluvLKK5Wdna2ff/7ZHisoKFCnTp0UFxdnb0tLS5PP59P27dvtmtTUVL9jpqWlqaCg4JR95OTkKCoqyl4SExMb5PoAAEBwCugTopNVV1drypQp6t27t6688kp7+2233aakpCQlJCToq6++UlZWlnbu3Km33npLkuT1ev3CkCR73ev1nrHG5/Ppl19+UUREhN9Ydna2MjMz7XWfz0coAgDgIhY0gSgjI0Pbtm3TRx995Ld9woQJ9p87deqk+Ph4DRgwQLt379YVV1xxTnpxOp1yOp3n5NgAACD4BMUrs0mTJmnVqlX64IMPdNlll52xtmfPnpKkXbt2SZLcbrdKSkr8amrWa+Ydna7G5XLVejoEAADME9BAZFmWJk2apBUrVmjDhg1KTk7+zX2KiookSfHx8ZIkj8ej4uJilZaW2jX5+flyuVxKSUmxa9avX+93nPz8fHk8nga6EgAAcCELaCDKyMjQf//3f2v58uVq1qyZvF6vvF6vfvnlF0nS7t27NXfuXBUWFuq7777TO++8ozFjxqhPnz7q3LmzJGngwIFKSUnRHXfcoa1bt2rt2rV64IEHlJGRYb/2mjhxov73f/9X06dP1zfffKPnn39er732mqZOnRqwawcAAMEjoIFo4cKFKi8vV9++fRUfH28vr776qiQpLCxM77//vgYOHKj27dvrvvvu04gRI/Tuu+/axwgJCdGqVasUEhIij8ej22+/XWPGjNGcOXPsmuTkZOXl5Sk/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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=df, x=\"InternetService\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>InternetService</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>DSL</th>\n",
       "      <td>0.189591</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fiber optic</th>\n",
       "      <td>0.418928</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    Churn\n",
       "InternetService          \n",
       "DSL              0.189591\n",
       "Fiber optic      0.418928\n",
       "No               0.074050"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Group by 'InternetService' and calculate the mean of 'Churn'\n",
    "df[['InternetService','Churn']].groupby('InternetService').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Internet service variable is definitely important in predicting churn rate.\n",
    "- As you can see, customers with **fiber optic internet service are much likely to churn than other customers** \n",
    "- Although there is not a big difference in the **number of customers** with **DSL** and **fiber optic.**\n",
    "- This company may have some problems with fiber optic connection. However, it is not a good way to make assumptions based on only one variable. Let's also check the monthly charges."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MonthlyCharges</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>InternetService</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>DSL</th>\n",
       "      <td>58.102169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fiber optic</th>\n",
       "      <td>91.500129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>21.079194</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 MonthlyCharges\n",
       "InternetService                \n",
       "DSL                   58.102169\n",
       "Fiber optic           91.500129\n",
       "No                    21.079194"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['InternetService','MonthlyCharges']].groupby('InternetService').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Fiber optic service is much more expensive than DSL which may be one of the reasons why customers churn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='TechSupport', ylabel='count'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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79q1a1/T8Ywx1n4l6bXXXivxHNcvfxFdi169eun777/X559/rtTUVN1///2Kj4/XzJkzdfz4cXXo0EGLFi0qsV3Dhg2vWP+jjz6q8ePH66uvvtLJkyeVk5Ojhx9+2GkN3t7eV6zxWuvA1ZGP5c/d8lEiI3+JjMTlkI/lj3w8j3ysXPlIo6CKef7559W2bVs1b97cWhYREaH169c7jFu/fr1uv/326/7BwEXt27fXRx99pMaNG1vhfymbzabOnTurc+fOmjhxosLDw7V48WIlJCSUWR3OvuZSxX3dPT09S3SX27dvr927d6tZs2ZldhxJCgwMVEhIiL777rvLdlgvdGevpePdsGFDxcbGKjY2Vvfcc4/Gjh2rmTNnqn379nrvvfcUEBAgX1/f66oxNDRU9957rxYtWqSTJ0+qe/fuCggIcDr2tttuk7e3t9LS0vT444+XWH8jdaAk8rHiuEs+SmSks3okMhKOyMeKQz5eRD46crd85NGDKiYyMlIxMTF6+eWXrWVPPvmk0tLSNHXqVH377bd666239Morr+ipp55yYaWVX3x8vI4cOaJHH31UW7ZsUVZWlpYvX64hQ4bo3Llz2rRpk5577jlt3bpVBw8e1Mcff6wffvhBERERZVqHs6+5VHFf98aNG2vt2rX673//a72xd/z48dqwYYNGjRqlzMxM7du3T59++qlGjRp1w8d75plnNH36dL388sv69ttvtXPnTiUlJenFF1+UdP4WP29vby1btkz5+fkqKChwup+JEyfq008/1f79+7Vr1y4tWbLE+trExMSoQYMG6tu3r9atW6fs7GytXr1aTzzxxDV97nFMTIzeffddffDBB5f9ZSSd/4ii8ePHa9y4cXr77beVlZWljRs36o033iiTOuCIfKw47pKPEhlJRuJakI8Vh3y8iHwsya3y8ZreZAC3denbL405/yIOT09Ph5eSfPjhh6Zly5amRo0aplGjRuZvf/tbBVdaNVz6opZvv/3W/O53vzP+/v7G29vbtGjRwowePdoUFxeb3bt3m+joaNOwYUPj5eVlbr/9djN37twbruFav+bGXP/X3dmLaC491v/93/+Ze++915pPT083rVu3Nl5eXg7H37x5s+nevbvx8fExtWvXNq1btzbTpk2z1jt7gU1SUpLx8/NzWHbpC3aMMWbRokWmbdu2xtPT09StW9d06dLFfPzxx9b61157zYSFhRkPDw+HWn9p6tSpJiIiwnh7e5t69eqZvn37mu+++85an5ubawYNGmQaNGhgvLy8zK233mqGDRtmCgoKLnttLjh69Kjx8vIytWrVKvGW4ku3O3funHn22WdNeHi49XV67rnnrrkOXB75WLHcIR+NISONISPJyKsjHysW+Xge+Vi58tFmzP//4AYAAAAAALjp8egBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKgHKWnJwsf39/V5cBAOWiqmTcgQMHZLPZlJmZ6epSALjApRmwevVq2Ww2HTt2zKV1VaSuXbtq9OjRri4DboJGAdzaDz/8oJEjR6pRo0by8vJSUFCQoqOjtX79ekmSzWbTJ5984toir+Lhhx/Wt99+W2b7u/CL7ErTrFmzVK1aNf33v/91uo/bbrtNCQkJZVYTgNIh4y6va9eustlsev7550us6927t2w2myZPnlxmxwsLC1Nubq7uuOOOMtsngIqRk5OjoUOHKiQkRJ6engoPD9f//d//6ccffyz1Pn/9618rNzdXfn5+ZViptGbNGnXr1k316tVTrVq1dNtttyk2NlanT58u0+OUxscff6ypU6da840bN9ZLL73kuoLgUjQK4Nb69++vbdu26a233tK3336rzz77TF27dr2u4Hd18Hp7eysgIKDM9nfh/8xemJ588km1atXKYdnw4cNVv359vfXWWyW2X7t2rfbv36+4uLgyqwlA6ZBxVxYWFqbk5GSHZf/973+Vlpam4ODgMj1WtWrVFBQUpOrVq5fpfgGUr++++0533nmn9u3bp3/+85/av3+/Fi5cqLS0NNntdh05cqRU+/X09FRQUJBsNluZ1bp792717NlTd955p9auXaudO3dq7ty58vT01Llz58rsONfrwu+RevXqqU6dOi6rA27GAG7q6NGjRpJZvXq10/Xh4eFGkjWFh4cbY4yZNGmSadOmjXnttddM48aNjc1ms/YXFxdnGjRoYOrUqWPuu+8+k5mZae1v//795re//a0JCAgwtWvXNnfeeadJTU0tccypU6eagQMHmtq1a5tGjRqZTz/91Bw+fNj89re/NbVr1zaRkZFmy5Yt1jZJSUnGz8/Pmr9Q39tvv23Cw8ONr6+vefjhh01hYaE1prCw0AwYMMDUqlXLBAUFmRdffNHce++95v/+7/9KXIcL+7tUQkKCue2220osj42NNR07dnR6TQFUHDLuyhl37733mpEjR5r69eubL7/80lo+bdo006dPH9OmTRszadIka/mRI0fMwIEDjb+/v/H29jY9e/Y03377rTHGmIKCAlOzZk3z+eefO5zvxx9/bHx8fMyJEydMdna2kWS2bdtmrd+5c6fp2bOnqV27tgkICDCPPfaY+eGHH6z1H3zwgbnjjjtMzZo1Tb169cz9999vjh8/7vTrCaB89OzZ04SGhpqff/7ZYXlubq6pVauWGTFihDHmfL5NmzbNDBkyxPj4+JiwsDDz6quvWuMvzYBVq1YZSebo0aPGmItZt2zZMtOiRQtTu3ZtEx0dbQ4dOuRw3Ndee820aNHCeHl5mebNm5t58+ZZ62bPnm0aN2581XNat26dufvuu03NmjVNaGio+fOf/+yQLadOnTLjxo0zoaGhxtPT0zRt2tS8/vrrDnX+0uLFi80v/7Pvcr9HfpnD9957r8PvIEnm+PHjpk6dOuaDDz4osf9atWo55DwqP+4ogNvy8fGRj4+PPvnkExUVFZVYv2XLFklSUlKScnNzrXlJ2r9/vz766CN9/PHH1rNmDz30kA4fPqx///vfysjIUPv27XX//fdbnebjx4/rgQceUFpamrZt26aePXuqT58+OnjwoMNxZ8+erc6dO2vbtm3q3bu3Bg4cqEGDBumxxx7TV199paZNm2rQoEEyxlz23LKysvTJJ59oyZIlWrJkidasWeNwe21CQoLWr1+vzz77TKmpqVq3bp2++uqr67p+cXFx2rdvn9auXWstO378uD788EPuJgDcABl39Yzz9PRUTEyMkpKSrGXJyckaOnRoibGDBw/W1q1b9dlnnyk9PV3GGD3wwAM6c+aMfH199Zvf/EYpKSkO2yxatEj9+vVTrVq1Suzv2LFj6tatm9q1a6etW7dq2bJlys/P1x/+8AdJUm5urh599FENHTpUe/bs0erVq/Xggw9e8boAKFtHjhzR8uXL9ac//Une3t4O64KCghQTE6P33nvP+rmcNWuW7rzzTm3btk1/+tOfNHLkSO3du/eaj/fzzz9r5syZ+sc//qG1a9fq4MGDeuqpp6z1ixYt0sSJEzVt2jTt2bNHzz33nJ5++mnrDs+goCDl5uY6/H+zS2VlZalnz57q37+/duzYoffee09ffvmlRo0aZY0ZNGiQ/vnPf+rll1/Wnj179Oqrr8rHx+eaz0Ny/nvklz7++GOFhoZqypQp1h2rtWvX1iOPPOKQydL531O///3vuRuhqnFpmwK4ig8//NDUrVvX1KxZ0/z61782iYmJZvv27dZ6SWbx4sUO20yaNMnUqFHDHD582Fq2bt064+vra06dOuUwtmnTpg7d5Eu1atXKzJ0715oPDw83jz32mDWfm5trJJmnn37aWpaenm4kmdzcXGOM87+2Xdp1HTt2rPVX/sLCQlOjRg2Hbu2xY8dMrVq1ruuOAmOM6dSpk4mNjbXm33jjDTq+gBsh485zlnEX/rKVmZlp6tSpY44fP27WrFljAgICzJkzZxzuKPj222+NJLN+/Xpr+//973/G29vbvP/++8aY83/xunD3gDEX7zL497//bYwp+dfEqVOnmh49ejhcr5ycHCPJ7N2712RkZBhJ5sCBA5e9vgDK18aNG53m5AUvvviikWTy8/NL5FtxcbEJCAgwCxYsMMZc2x0Fksz+/futfcybN88EBgZa802bNjUpKSkONUydOtXY7XZjjDFnz541gwcPNpJMUFCQ6devn5k7d64pKCiwxsfFxZnhw4c77GPdunXGw8PDnDx50uzdu9dIKnFH2AXXekfBpb9HjDEl7uwKDw83s2fPdhizadMmU61aNetOivz8fFO9evXL3h2Hyos7CuDW+vfvr0OHDumzzz5Tz549tXr1arVv377EM6uXCg8PV8OGDa357du36/jx46pfv771VzwfHx9lZ2crKytL0vm/tj311FOKiIiQv7+/fHx8tGfPnhJ/bWvdurX178DAQElSZGRkiWWHDx++bH2NGzd26LoGBwdb47/77judOXNGd911l7Xez89PzZs3v+I5OzN06FB9+OGH+umnnyRJb775ph566CE6voCbIOPOu1LGtWnTRrfddps+/PBDvfnmmxo4cGCJ9wjs2bNH1atXV8eOHa1l9evXV/PmzbVnzx5J0gMPPKAaNWros88+kyR99NFH8vX1VVRUlNPjbt++XatWrXK4ni1atJB0/i9+bdq00f3336/IyEg99NBDeu2113T06NHLXhMA5cdc4508v8w3m82moKCgK2bZpWrVqqWmTZta87/MthMnTigrK0txcXEOufHss89aOVytWjUlJSXpP//5j2bMmKFbbrlFzz33nPWuKel89iQnJzvsIzo6WsXFxcrOzlZmZqaqVaume++995rrdubS3yPX6q677lKrVq2suyTeeecdhYeHq0uXLjdUD9wPb+yB26tZs6a6d++u7t276+mnn9bjjz+uSZMmafDgwZfdpnbt2g7zx48fV3BwsFavXl1i7IWP9XrqqaeUmpqqmTNnqlmzZvL29tbvf//7Ei8Kq1GjhvXvCy+4cbasuLj4svX9cvyFba40vrQeeeQRjRkzRu+//766dOmi9evXa/r06WV+HAClR8Zd3dChQzVv3jzt3r1bmzdvLtU+PD099fvf/14pKSl65JFHlJKSoocffviyLy88fvy4+vTpoxdeeKHEuuDgYFWrVk2pqanasGGDVqxYoblz5+qvf/2rNm3apCZNmpSqRgDXp1mzZrLZbNqzZ49+97vflVi/Z88e1a1b1/oP4hvNJmfbX2hSHD9+XJL02muvOTQtpfMNgl+65ZZbNHDgQA0cOFBTp07V7bffroULF+qZZ57R8ePH9cc//lFPPPFEieM3atRI+/fvv2KNHh4eJRonZ86cKTHu0t8j1+Pxxx/XvHnz9P/+3/9TUlKShgwZUqYvfYR74I4CVDotW7bUiRMnJJ0P7Gt5S2z79u2Vl5en6tWrq1mzZg5TgwYNJEnr16/X4MGD9bvf/U6RkZEKCgrSgQMHyvNUnLr11ltVo0YNh+eRCwoKSvXxY3Xq1NFDDz2kN998U0lJSbr99tt1zz33lGW5AMoYGVfSgAEDtHPnTt1xxx1q2bJlifURERE6e/asNm3aZC378ccftXfvXofxMTExWrZsmXbt2qUvvvhCMTExlz1m+/bttWvXLjVu3LjENb3wf7BtNps6d+6sZ555Rtu2bZOnp6cWL158XdcDQOnVr19f3bt31/z583Xy5EmHdXl5eVq0aJEefvjhCvmP2MDAQIWEhOi7774rkRlXah7WrVtXwcHBVu63b99eu3fvLrGPZs2aydPTU5GRkSouLtaaNWuc7q9hw4b66aefrP1JcvoOgmtxuU9jeOyxx/T999/r5Zdf1u7duxUbG1uq/cO90SiA2/rxxx/VrVs3vfPOO9qxY4eys7P1wQcfaMaMGerbt6+k87e3pqWlKS8v74q3fEZFRclut6tfv35asWKFDhw4oA0bNuivf/2rtm7dKkm67bbbrBe6bN++XQMGDCiXv/JfTZ06dRQbG6uxY8dq1apV2rVrl+Li4uTh4VGqX3RxcXHasGGDFi5c6PQFYABcg4y79oyrW7eucnNzlZaW5nT9bbfdpr59+2rYsGH68ssvtX37dj322GO65ZZbrGspSV26dLFecNakSZMSf/X7pfj4eB05ckSPPvqotmzZoqysLC1fvlxDhgzRuXPntGnTJj333HPaunWrDh48qI8//lg//PCDIiIibuwCAbgur7zyioqKihQdHa21a9cqJydHy5YtU/fu3XXLLbdo2rRpFVbLM888o+nTp+vll1/Wt99+q507dyopKUkvvviiJOnVV1/VyJEjtWLFCmVlZWnXrl0aP368du3apT59+kiSxo8frw0bNmjUqFHKzMzUvn379Omnn1ovM2zcuLFiY2M1dOhQffLJJ8rOztbq1av1/vvvS5I6duyoWrVq6S9/+YuysrKUkpJy1cfZLqdx48Zau3at/vvf/+p///uftbxu3bp68MEHNXbsWPXo0UOhoaE3cNXgrmgUwG35+PioY8eOmj17trp06aI77rhDTz/9tIYNG6ZXXnlF0vm316ampiosLEzt2rW77L5sNps+//xzdenSRUOGDNHtt9+uRx55RN9//731vO2LL76ounXr6te//rX69Omj6OhotW/fvkLO9VIvvvii7Ha7fvOb3ygqKkqdO3dWRESEatased37uvvuu9W8eXMVFhZq0KBB5VAtgNIg464v4/z9/a94q2xSUpI6dOig3/zmN7Lb7TLG6PPPPy/x2MSjjz6q7du3X/FuAkkKCQnR+vXrde7cOfXo0UORkZEaPXq0/P395eHhIV9fX61du1YPPPCAbr/9dk2YMEGzZs1Sr169rv+CACi12267TVu3btWtt96qP/zhD2ratKmGDx+u++67T+np6apXr16F1fL444/r9ddfV1JSkiIjI3XvvfcqOTnZuqPgrrvu0vHjxzVixAi1atVK9957rzZu3KhPPvnEeudA69attWbNGn377be655571K5dO02cOFEhISHWcRYsWKDf//73+tOf/qQWLVpo2LBh1h0E9erV0zvvvKPPP/9ckZGR+uc//6nJkyeX6nymTJmiAwcOqGnTpiXeZxAXF6fTp0/zR6gqzGau9e0fAFzmxIkTuuWWWzRr1iw+2hBAlUPGAUDl8o9//ENjxozRoUOH5Onp6epyUA54mSHghrZt26ZvvvlGd911lwoKCjRlyhRJcriFFgAqKzIOACqnn3/+Wbm5uXr++ef1xz/+kSZBFcajB4Cbmjlzptq0aaOoqCidOHFC69ats15KBgCVHRkHAJXPjBkz1KJFCwUFBSkxMdHV5aAc8egBAAAAAACwcEcBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFj4eMRrUFxcrEOHDqlOnTqy2WyuLgdAJWSM0U8//aSQkBB5eFStHi0ZCeBGkI8A4Jwr85FGwTU4dOiQwsLCXF0GgCogJydHoaGhri6jTJGRAMoC+QgAzrkiH2kUXIM6depIOv8F8vX1dXE1ACqjwsJChYWFWXlSlZCRAG4E+QgAzrkyH2kUXIMLt4r5+voS8gBuSFW89ZSMBFAWyEcAcM4V+Vi1HgQDAAAAAAA3hEYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMDi0kbBggUL1Lp1a+sFL3a7Xf/+97+t9V27dpXNZnOYRowY4bCPgwcPqnfv3qpVq5YCAgI0duxYnT171mHM6tWr1b59e3l5ealZs2ZKTk6uiNMDAAAAAKDScemnHoSGhur555/XbbfdJmOM3nrrLfXt21fbtm1Tq1atJEnDhg3TlClTrG1q1apl/fvcuXPq3bu3goKCtGHDBuXm5mrQoEGqUaOGnnvuOUlSdna2evfurREjRmjRokVKS0vT448/ruDgYEVHR1fsCQMAAAAA4OZc2ijo06ePw/y0adO0YMECbdy40WoU1KpVS0FBQU63X7FihXbv3q2VK1cqMDBQbdu21dSpUzV+/HhNnjxZnp6eWrhwoZo0aaJZs2ZJkiIiIvTll19q9uzZNAoAAAAAALiE27yj4Ny5c3r33Xd14sQJ2e12a/miRYvUoEED3XHHHUpMTNTPP/9srUtPT1dkZKQCAwOtZdHR0SosLNSuXbusMVFRUQ7Hio6OVnp6+mVrKSoqUmFhocMEADiPjAQA58hHAFWFyxsFO3fulI+Pj7y8vDRixAgtXrxYLVu2lCQNGDBA77zzjlatWqXExET94x//0GOPPWZtm5eX59AkkGTN5+XlXXFMYWGhTp486bSm6dOny8/Pz5rCwsLK7HwBoLIjIwHAOfIRQFXh8kZB8+bNlZmZqU2bNmnkyJGKjY3V7t27JUnDhw9XdHS0IiMjFRMTo7fffluLFy9WVlZWudaUmJiogoICa8rJySnX4wFAZUJGAoBz5COAqsKl7yiQJE9PTzVr1kyS1KFDB23ZskVz5szRq6++WmJsx44dJUn79+9X06ZNFRQUpM2bNzuMyc/PlyTrvQZBQUHWsl+O8fX1lbe3t9OavLy85OXldWMnBgBVFBkJAM6VNh87jH27HKqpWjL+NsjVJQA3FZffUXCp4uJiFRUVOV2XmZkpSQoODpYk2e127dy5U4cPH7bGpKamytfX13p8wW63Ky0tzWE/qampDu9BAAAAAAAA57n0joLExET16tVLjRo10k8//aSUlBStXr1ay5cvV1ZWllJSUvTAAw+ofv362rFjh8aMGaMuXbqodevWkqQePXqoZcuWGjhwoGbMmKG8vDxNmDBB8fHxVjd3xIgReuWVVzRu3DgNHTpUX3zxhd5//30tXbrUlacOAAAAAIBbcmmj4PDhwxo0aJByc3Pl5+en1q1ba/ny5erevbtycnK0cuVKvfTSSzpx4oTCwsLUv39/TZgwwdq+WrVqWrJkiUaOHCm73a7atWsrNjZWU6ZMscY0adJES5cu1ZgxYzRnzhyFhobq9ddf56MRAQAAAABwwqWNgjfeeOOy68LCwrRmzZqr7iM8PFyff/75Fcd07dpV27Ztu+76AAAAAAC42bjdOwoAAAAAAIDr0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFiqu7oAAMDNocPYt11dgtvL+NsgV5cAAABAowCV18Epka4uwe01mrjT1SUAAAAAqGRc+ujBggUL1Lp1a/n6+srX11d2u13//ve/rfWnTp1SfHy86tevLx8fH/Xv31/5+fkO+zh48KB69+6tWrVqKSAgQGPHjtXZs2cdxqxevVrt27eXl5eXmjVrpuTk5Io4PQAAAAAAKh2XNgpCQ0P1/PPPKyMjQ1u3blW3bt3Ut29f7dq1S5I0ZswY/etf/9IHH3ygNWvW6NChQ3rwwQet7c+dO6fevXvr9OnT2rBhg9566y0lJydr4sSJ1pjs7Gz17t1b9913nzIzMzV69Gg9/vjjWr58eYWfLwAAAAAA7s6ljx706dPHYX7atGlasGCBNm7cqNDQUL3xxhtKSUlRt27dJElJSUmKiIjQxo0b1alTJ61YsUK7d+/WypUrFRgYqLZt22rq1KkaP368Jk+eLE9PTy1cuFBNmjTRrFmzJEkRERH68ssvNXv2bEVHR1f4OQMAAAAA4M7c5lMPzp07p3fffVcnTpyQ3W5XRkaGzpw5o6ioKGtMixYt1KhRI6Wnp0uS0tPTFRkZqcDAQGtMdHS0CgsLrbsS0tPTHfZxYcyFfQAAAAAAgItc/jLDnTt3ym6369SpU/Lx8dHixYvVsmVLZWZmytPTU/7+/g7jAwMDlZeXJ0nKy8tzaBJcWH9h3ZXGFBYW6uTJk/L29i5RU1FRkYqKiqz5wsLCGz5PAKgqyEgAcI58BFBVuPyOgubNmyszM1ObNm3SyJEjFRsbq927d7u0punTp8vPz8+awsLCXFoPALgTMhIAnCMfAVQVLm8UeHp6qlmzZurQoYOmT5+uNm3aaM6cOQoKCtLp06d17Ngxh/H5+fkKCgqSJAUFBZX4FIQL81cb4+vr6/RuAklKTExUQUGBNeXk5JTFqQJAlUBGAoBz5COAqsLljYJLFRcXq6ioSB06dFCNGjWUlpZmrdu7d68OHjwou90uSbLb7dq5c6cOHz5sjUlNTZWvr69atmxpjfnlPi6MubAPZ7y8vKyPbLwwAQDOIyMBwDnyEUBV4dJ3FCQmJqpXr15q1KiRfvrpJ6WkpGj16tVavny5/Pz8FBcXp4SEBNWrV0++vr7685//LLvdrk6dOkmSevTooZYtW2rgwIGaMWOG8vLyNGHCBMXHx8vLy0uSNGLECL3yyisaN26chg4dqi+++ELvv/++li5d6spTBwAAAADALbm0UXD48GENGjRIubm58vPzU+vWrbV8+XJ1795dkjR79mx5eHiof//+KioqUnR0tObPn29tX61aNS1ZskQjR46U3W5X7dq1FRsbqylTplhjmjRpoqVLl2rMmDGaM2eOQkND9frrr/PRiAAAAAAAOOHSRsEbb7xxxfU1a9bUvHnzNG/evMuOCQ8P1+eff37F/XTt2lXbtm0rVY0AAAAAANxM3O4dBQAAAAAAwHVoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWlzYKpk+frl/96leqU6eOAgIC1K9fP+3du9dhTNeuXWWz2RymESNGOIw5ePCgevfurVq1aikgIEBjx47V2bNnHcasXr1a7du3l5eXl5o1a6bk5OTyPj0AAAAAACodlzYK1qxZo/j4eG3cuFGpqak6c+aMevTooRMnTjiMGzZsmHJzc61pxowZ1rpz586pd+/eOn36tDZs2KC33npLycnJmjhxojUmOztbvXv31n333afMzEyNHj1ajz/+uJYvX15h5woAAAAAQGVQ3ZUHX7ZsmcN8cnKyAgIClJGRoS5duljLa9WqpaCgIKf7WLFihXbv3q2VK1cqMDBQbdu21dSpUzV+/HhNnjxZnp6eWrhwoZo0aaJZs2ZJkiIiIvTll19q9uzZio6OLr8TBAAAAACgknGrdxQUFBRIkurVq+ewfNGiRWrQoIHuuOMOJSYm6ueff7bWpaenKzIyUoGBgday6OhoFRYWateuXdaYqKgoh31GR0crPT3daR1FRUUqLCx0mAAA55GRAOAc+QigqnCbRkFxcbFGjx6tzp0764477rCWDxgwQO+8845WrVqlxMRE/eMf/9Bjjz1mrc/Ly3NoEkiy5vPy8q44prCwUCdPnixRy/Tp0+Xn52dNYWFhZXaeAFDZkZEA4Bz5CKCqcJtGQXx8vL7++mu9++67DsuHDx+u6OhoRUZGKiYmRm+//bYWL16srKyscqslMTFRBQUF1pSTk1NuxwKAyoaMBADnyEcAVYVL31FwwahRo7RkyRKtXbtWoaGhVxzbsWNHSdL+/fvVtGlTBQUFafPmzQ5j8vPzJcl6r0FQUJC17JdjfH195e3tXeIYXl5e8vLyKvX5AEBVRkYCgHPkI4CqwqV3FBhjNGrUKC1evFhffPGFmjRpctVtMjMzJUnBwcGSJLvdrp07d+rw4cPWmNTUVPn6+qply5bWmLS0NIf9pKamym63l9GZAAAAAABQNbi0URAfH6933nlHKSkpqlOnjvLy8pSXl2e9NyArK0tTp05VRkaGDhw4oM8++0yDBg1Sly5d1Lp1a0lSjx491LJlSw0cOFDbt2/X8uXLNWHCBMXHx1sd3REjRui7777TuHHj9M0332j+/Pl6//33NWbMGJedOwAAAAAA7siljx4sWLBAktS1a1eH5UlJSRo8eLA8PT21cuVKvfTSSzpx4oTCwsLUv39/TZgwwRpbrVo1LVmyRCNHjpTdblft2rUVGxurKVOmWGOaNGmipUuXasyYMZozZ45CQ0P1+uuvl8tHI3YY+3aZ77MqyfjbIFeXAABV3sEpka4uwe01mrjT1SUAAOC2XNooMMZccX1YWJjWrFlz1f2Eh4fr888/v+KYrl27atu2bddVHwAAAAAANxu3+dQDAAAAAADgejQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAACWUjUKunXrpmPHjpVYXlhYqG7dut1oTQCAckJ+A4Bz5CMAXFSqRsHq1at1+vTpEstPnTqldevW3XBRAIDyQX4DgHPkIwBcVP16Bu/YscP69+7du5WXl2fNnzt3TsuWLdMtt9xyzfubPn26Pv74Y33zzTfy9vbWr3/9a73wwgtq3ry5NebUqVN68skn9e6776qoqEjR0dGaP3++AgMDrTEHDx7UyJEjtWrVKvn4+Cg2NlbTp09X9eoXT2/16tVKSEjQrl27FBYWpgkTJmjw4MHXc/oAUGmVdX4DQFVBPgJASdfVKGjbtq1sNptsNpvTW7C8vb01d+7ca97fmjVrFB8fr1/96lc6e/as/vKXv6hHjx7avXu3ateuLUkaM2aMli5dqg8++EB+fn4aNWqUHnzwQa1fv17S+QDv3bu3goKCtGHDBuXm5mrQoEGqUaOGnnvuOUlSdna2evfurREjRmjRokVKS0vT448/ruDgYEVHR1/PJQCASqms8xsAqgry8eZycEqkq0twe40m7nR1CXAD19UoyM7OljFGt956qzZv3qyGDRta6zw9PRUQEKBq1apd8/6WLVvmMJ+cnKyAgABlZGSoS5cuKigo0BtvvKGUlBQruJOSkhQREaGNGzeqU6dOWrFihXbv3q2VK1cqMDBQbdu21dSpUzV+/HhNnjxZnp6eWrhwoZo0aaJZs2ZJkiIiIvTll19q9uzZNAoA3BTKOr8BoKogHwGgpOtqFISHh0uSiouLy6WYgoICSVK9evUkSRkZGTpz5oyioqKsMS1atFCjRo2Unp6uTp06KT09XZGRkQ6PIkRHR2vkyJHatWuX2rVrp/T0dId9XBgzevRop3UUFRWpqKjImi8sLCyrUwQAlyjL/CYjAVQl5CMAlHRdjYJf2rdvn1atWqXDhw+XCNaJEyde9/6Ki4s1evRode7cWXfccYckKS8vT56envL393cYGxgYaD0/lpeX59AkuLD+wrorjSksLNTJkyfl7e3tsG769Ol65plnrvscAKAyuNH8JiMBVFXkIwCcV6pGwWuvvaaRI0eqQYMGCgoKks1ms9bZbLZSNQri4+P19ddf68svvyxNSWUqMTFRCQkJ1nxhYaHCwsJcWBEAlI2yyG8yEkBVRD4CwEWlahQ8++yzmjZtmsaPH18mRYwaNUpLlizR2rVrFRoaai0PCgrS6dOndezYMYe7CvLz8xUUFGSN2bx5s8P+8vPzrXUX/vfCsl+O8fX1LXE3gSR5eXnJy8urTM4NANxJWeQ3GQmgKiIfAeAij9JsdPToUT300EM3fHBjjEaNGqXFixfriy++UJMmTRzWd+jQQTVq1FBaWpq1bO/evTp48KDsdrskyW63a+fOnTp8+LA1JjU1Vb6+vmrZsqU15pf7uDDmwj4A4GZRVvkNAFUN+QgAF5WqUfDQQw9pxYoVN3zw+Ph4vfPOO0pJSVGdOnWUl5envLw8nTx5UpLk5+enuLg4JSQkaNWqVcrIyNCQIUNkt9vVqVMnSVKPHj3UsmVLDRw4UNu3b9fy5cs1YcIExcfHWx3dESNG6LvvvtO4ceP0zTffaP78+Xr//fc1ZsyYGz4HAKhMyiq/AaCqIR8B4KJSPXrQrFkzPf3009q4caMiIyNVo0YNh/VPPPHENe1nwYIFkqSuXbs6LE9KStLgwYMlSbNnz5aHh4f69++voqIiRUdHa/78+dbYatWqacmSJRo5cqTsdrtq166t2NhYTZkyxRrTpEkTLV26VGPGjNGcOXMUGhqq119/nY9GBHDTKav8BoCqhnwEgItsxhhzvRtd+oiAww5tNn333Xc3VJS7KSwslJ+fnwoKCuTr63vFsR3Gvl1BVVVOGX8bVGb7Ojglssz2VVU1mrjT1SXg/3c9OVKeyiO/r/XcyMerK6uMJB+vjnx0H+Qj+XgtyMeKQz66D1fmY6nuKMjOzi7rOgAAFYD8BgDnyEcAuKhU7ygAAAAAAABVU6nuKBg6dOgV17/55pulKgYAUL7IbwBwjnwEgItK1Sg4evSow/yZM2f09ddf69ixY+rWrVuZFAYAKHvkNwA4Rz4CwEWlahQsXry4xLLi4mKNHDlSTZs2veGiAADlg/wGAOfIRwC4qMzeUeDh4aGEhATNnj27rHYJAKgA5DcAOEc+ArhZlenLDLOysnT27Nmy3CUAoAKQ3wDgHPkI4GZUqkcPEhISHOaNMcrNzdXSpUsVGxtbJoUBAMoe+Q0AzpGPAHBRqRoF27Ztc5j38PBQw4YNNWvWrKu+MRYA4DrkNwA4Rz4CwEWlahSsWrWqrOsAAFQA8hsAnCMfAeCiUjUKLvjhhx+0d+9eSVLz5s3VsGHDMikKAFC+yG8AcI58BIBSvszwxIkTGjp0qIKDg9WlSxd16dJFISEhiouL088//1zWNQIAygj5DQDOkY8AcFGpGgUJCQlas2aN/vWvf+nYsWM6duyYPv30U61Zs0ZPPvlkWdcIACgj5DcAOEc+AsBFpXr04KOPPtKHH36orl27WsseeOABeXt76w9/+IMWLFhQVvUBAMoQ+Q0AzpGPAHBRqe4o+PnnnxUYGFhieUBAALdmAYAbI78BwDnyEQAuKlWjwG63a9KkSTp16pS17OTJk3rmmWdkt9vLrDgAQNkivwHAOfIRAC4q1aMHL730knr27KnQ0FC1adNGkrR9+3Z5eXlpxYoVZVogAKDskN8A4Bz5CAAXlapREBkZqX379mnRokX65ptvJEmPPvqoYmJi5O3tXaYFAgDKDvkNAM6RjwBwUakaBdOnT1dgYKCGDRvmsPzNN9/UDz/8oPHjx5dJcQCAskV+A4Bz5CMAXFSqdxS8+uqratGiRYnlrVq10sKFC2+4KABA+SC/AcA58hEALipVoyAvL0/BwcElljds2FC5ubk3XBQAoHyQ3wDgHPkIABeVqlEQFham9evXl1i+fv16hYSE3HBRAIDyQX4DgHPkIwBcVKpGwbBhwzR69GglJSXp+++/1/fff68333xTY8aMKfFc15WsXbtWffr0UUhIiGw2mz755BOH9YMHD5bNZnOYevbs6TDmyJEjiomJka+vr/z9/RUXF6fjx487jNmxY4fuuece1axZU2FhYZoxY0ZpThsAKr2yym8AqGrIRwC4qFQvMxw7dqx+/PFH/elPf9Lp06clSTVr1tT48eOVmJh4zfs5ceKE2rRpo6FDh+rBBx90OqZnz55KSkqy5r28vBzWx8TEKDc3V6mpqTpz5oyGDBmi4cOHKyUlRZJUWFioHj16KCoqSgsXLtTOnTs1dOhQ+fv7a/jw4dd76gBQqZVVfgNAVUM+AsBFpWoU2Gw2vfDCC3r66ae1Z88eeXt767bbbivxH/FX06tXL/Xq1euKY7y8vBQUFOR03Z49e7Rs2TJt2bJFd955pyRp7ty5euCBBzRz5kyFhIRo0aJFOn36tN588015enqqVatWyszM1IsvvkijAMBNp6zyGwCqGvIRAC4qVaPgAh8fH/3qV78qq1qcWr16tQICAlS3bl1169ZNzz77rOrXry9JSk9Pl7+/v9UkkKSoqCh5eHho06ZN+t3vfqf09HR16dJFnp6e1pjo6Gi98MILOnr0qOrWrVuu9QNVQee5nV1dgttb/+eSz7W6s4rIb+BmQD5eHfkI3JzIx6tz53y8oUZBeevZs6cefPBBNWnSRFlZWfrLX/6iXr16KT09XdWqVVNeXp4CAgIctqlevbrq1aunvLw8SeffYNukSROHMYGBgdY6Z42CoqIiFRUVWfOFhYVlfWoAUGmRkQDgHPkIoKoo1csMK8ojjzyi3/72t4qMjFS/fv20ZMkSbdmyRatXry7X406fPl1+fn7WFBYWVq7HA4DKhIwEAOfIRwBVhVs3Ci516623qkGDBtq/f78kKSgoSIcPH3YYc/bsWR05csR6r0FQUJDy8/MdxlyYv9y7DxITE1VQUGBNOTk5ZX0qAFBpkZEA4Bz5CKCqcOtHDy71n//8Rz/++KOCg4MlSXa7XceOHVNGRoY6dOggSfriiy9UXFysjh07WmP++te/6syZM6pRo4YkKTU1Vc2bN7/s+wm8vLx4cQ0AXAYZCQDOkY8AqgqX3lFw/PhxZWZmKjMzU5KUnZ2tzMxMHTx4UMePH9fYsWO1ceNGHThwQGlpaerbt6+aNWum6OhoSVJERIR69uypYcOGafPmzVq/fr1GjRqlRx55RCEhIZKkAQMGyNPTU3Fxcdq1a5fee+89zZkzRwkJCa46bQAAAAAA3JZLGwVbt25Vu3bt1K5dO0lSQkKC2rVrp4kTJ6patWrasWOHfvvb3+r2229XXFycOnTooHXr1jl0ahctWqQWLVro/vvv1wMPPKC7775bf//73631fn5+WrFihbKzs9WhQwc9+eSTmjhxIh+NCAAAAACAEy599KBr164yxlx2/fLly6+6j3r16iklJeWKY1q3bq1169Zdd30AAAAAANxsKtXLDAEAAAAAQPmiUQAAAAAAACw0CgAAAAAAgIVGAQAAAAAAsNAoAAAAAAAAFhoFAAAAAADAQqMAAAAAAABYaBQAAAAAAAALjQIAAAAAAGChUQAAAAAAACw0CgAAAAAAgIVGAQAAAAAAsNAoAAAAAAAAFhoFAAAAAADAQqMAAAAAAABYaBQAAAAAAAALjQIAAAAAAGChUQAAAAAAACw0CgAAAAAAgIVGAQAAAAAAsNAoAAAAAAAAFhoFAAAAAADAQqMAAAAAAABYXNooWLt2rfr06aOQkBDZbDZ98sknDuuNMZo4caKCg4Pl7e2tqKgo7du3z2HMkSNHFBMTI19fX/n7+ysuLk7Hjx93GLNjxw7dc889qlmzpsLCwjRjxozyPjUAAAAAACollzYKTpw4oTZt2mjevHlO18+YMUMvv/yyFi5cqE2bNql27dqKjo7WqVOnrDExMTHatWuXUlNTtWTJEq1du1bDhw+31hcWFqpHjx4KDw9XRkaG/va3v2ny5Mn6+9//Xu7nBwAAAABAZVPdlQfv1auXevXq5XSdMUYvvfSSJkyYoL59+0qS3n77bQUGBuqTTz7RI488oj179mjZsmXasmWL7rzzTknS3Llz9cADD2jmzJkKCQnRokWLdPr0ab355pvy9PRUq1atlJmZqRdffNGhoQAAAAAAANz4HQXZ2dnKy8tTVFSUtczPz08dO3ZUenq6JCk9PV3+/v5Wk0CSoqKi5OHhoU2bNlljunTpIk9PT2tMdHS09u7dq6NHjzo9dlFRkQoLCx0mAMB5ZCQAOEc+Aqgq3LZRkJeXJ0kKDAx0WB4YGGity8vLU0BAgMP66tWrq169eg5jnO3jl8e41PTp0+Xn52dNYWFhN35CAFBFkJEA4Bz5CKCqcNtGgSslJiaqoKDAmnJyclxdEgC4DTISAJwjHwFUFS59R8GVBAUFSZLy8/MVHBxsLc/Pz1fbtm2tMYcPH3bY7uzZszpy5Ii1fVBQkPLz8x3GXJi/MOZSXl5e8vLyKpPzAICqhowEAOfIRwBVhdveUdCkSRMFBQUpLS3NWlZYWKhNmzbJbrdLkux2u44dO6aMjAxrzBdffKHi4mJ17NjRGrN27VqdOXPGGpOamqrmzZurbt26FXQ2AAAAAABUDi5tFBw/flyZmZnKzMyUdP4FhpmZmTp48KBsNptGjx6tZ599Vp999pl27typQYMGKSQkRP369ZMkRUREqGfPnho2bJg2b96s9evXa9SoUXrkkUcUEhIiSRowYIA8PT0VFxenXbt26b333tOcOXOUkJDgorMGAAAAAMB9ufTRg61bt+q+++6z5i/8x3tsbKySk5M1btw4nThxQsOHD9exY8d09913a9myZapZs6a1zaJFizRq1Cjdf//98vDwUP/+/fXyyy9b6/38/LRixQrFx8erQ4cOatCggSZOnMhHIwIAAAAA4IRLGwVdu3aVMeay6202m6ZMmaIpU6Zcdky9evWUkpJyxeO0bt1a69atK3WdAAAAAADcLNz2HQUAAAAAAKDi0SgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYHHrRsHkyZNls9kcphYtWljrT506pfj4eNWvX18+Pj7q37+/8vPzHfZx8OBB9e7dW7Vq1VJAQIDGjh2rs2fPVvSpAAAAAABQKVR3dQFX06pVK61cudKar179YsljxozR0qVL9cEHH8jPz0+jRo3Sgw8+qPXr10uSzp07p969eysoKEgbNmxQbm6uBg0apBo1aui5556r8HMBAAAAAMDduX2joHr16goKCiqxvKCgQG+88YZSUlLUrVs3SVJSUpIiIiK0ceNGderUSStWrNDu3bu1cuVKBQYGqm3btpo6darGjx+vyZMny9PTs6JPBwAAAAAAt+bWjx5I0r59+xQSEqJbb71VMTExOnjwoCQpIyNDZ86cUVRUlDW2RYsWatSokdLT0yVJ6enpioyMVGBgoDUmOjpahYWF2rVr12WPWVRUpMLCQocJAHAeGQkAzpGPAKoKt24UdOzYUcnJyVq2bJkWLFig7Oxs3XPPPfrpp5+Ul5cnT09P+fv7O2wTGBiovLw8SVJeXp5Dk+DC+gvrLmf69Ony8/OzprCwsLI9MQCoxMhIAHCOfARQVbh1o6BXr1566KGH1Lp1a0VHR+vzzz/XsWPH9P7775frcRMTE1VQUGBNOTk55Xo8AKhMyEgAcI58BFBVuP07Cn7J399ft99+u/bv36/u3bvr9OnTOnbsmMNdBfn5+dY7DYKCgrR582aHfVz4VARn7z24wMvLS15eXmV/AgBQBZCRAOAc+QigqnDrOwoudfz4cWVlZSk4OFgdOnRQjRo1lJaWZq3fu3evDh48KLvdLkmy2+3auXOnDh8+bI1JTU2Vr6+vWrZsWeH1AwAAAADg7tz6joKnnnpKffr0UXh4uA4dOqRJkyapWrVqevTRR+Xn56e4uDglJCSoXr168vX11Z///GfZ7XZ16tRJktSjRw+1bNlSAwcO1IwZM5SXl6cJEyYoPj6ebi8AAAAAAE64daPgP//5jx599FH9+OOPatiwoe6++25t3LhRDRs2lCTNnj1bHh4e6t+/v4qKihQdHa358+db21erVk1LlizRyJEjZbfbVbt2bcXGxmrKlCmuOiUAAAAAANyaWzcK3n333Suur1mzpubNm6d58+Zddkx4eLg+//zzsi4NAAAAAIAqqVK9owAAAAAAAJQvGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDQKAAAAAACAhUYBAAAAAACw0CgAAAAAAAAWGgUAAAAAAMBCowAAAAAAAFhoFAAAAAAAAAuNAgAAAAAAYKFRAAAAAAAALDdVo2DevHlq3LixatasqY4dO2rz5s2uLgkAAAAAALdy0zQK3nvvPSUkJGjSpEn66quv1KZNG0VHR+vw4cOuLg0AAAAAALdx0zQKXnzxRQ0bNkxDhgxRy5YttXDhQtWqVUtvvvmmq0sDAAAAAMBt3BSNgtOnTysjI0NRUVHWMg8PD0VFRSk9Pd2FlQEAAAAA4F6qu7qAivC///1P586dU2BgoMPywMBAffPNNyXGFxUVqaioyJovKCiQJBUWFl71WOeKTt5gtVXbtVzDa/XTqXNltq+qqqyu99mTZ8tkP1XZ1a71hfXGmIoop1yVNiPJx6srq59Z8vHqyMeKQz6Sj2WBfKw45GPFcet8NDeB//73v0aS2bBhg8PysWPHmrvuuqvE+EmTJhlJTExMTGU+5eTkVFT0lRsykomJqTwm8pGJiYnJ+eSKfLQZUwXat1dx+vRp1apVSx9++KH69etnLY+NjdWxY8f06aefOoy/tBtcXFysI0eOqH79+rLZbBVV9g0rLCxUWFiYcnJy5Ovr6+pyqjSudcWprNfaGKOffvpJISEh8vCo3E99VYWMrKzfR5UR17piVcbrTT66l8r4PVSZcb0rTmW81q7Mx5vi0QNPT0916NBBaWlpVqOguLhYaWlpGjVqVInxXl5e8vLycljm7+9fAZWWD19f30rzw1DZca0rTmW81n5+fq4uoUxUpYysjN9HlRXXumJVtutNPrqfyvY9VNlxvStOZbvWrsrHm6JRIEkJCQmKjY3VnXfeqbvuuksvvfSSTpw4oSFDhri6NAAAAAAA3MZN0yh4+OGH9cMPP2jixInKy8tT27ZttWzZshIvOAQAAAAA4GZ20zQKJGnUqFFOHzWoqry8vDRp0qQSt8Ch7HGtKw7XGmWB76OKw7WuWFxv3Ci+hyoW17vicK2vz03xMkMAAAAAAHBtKverZQEAAAAAQJmiUQAAAAAAACw0CgAAAAAAgIVGAQDL5MmT1bZtW1eXUaUNHjxY/fr1c3UZAEqBjCx/ZCRQOZGP5a+i85FGQSVkjFFUVJSio6NLrJs/f778/f31n//8xwWVVW2DBw+WzWbT888/77D8k08+kc1mqxJ1PPXUU0pLS7uubRo3bqyXXnrpho5blg4cOCCbzabMzExXl+LUnDlzlJyc7Ooybgru8jNbVbnr7yIy8srISEjkY3kjH68N+Xh9KjofaRRUQjabTUlJSdq0aZNeffVVa3l2drbGjRunuXPnKjQ01IUVVl01a9bUCy+8oKNHj1bJOnx8fFS/fv0y3ee1On36tEuOW1autX4/Pz/5+/uXbzGwuMvPbFXkzr+LyEj3Q0a6H/Kx/JCPFYt8LB80CiqpsLAwzZkzR0899ZSys7NljFFcXJx69Oihdu3aqVevXvLx8VFgYKAGDhyo//3vf9a2H374oSIjI+Xt7a369esrKipKJ06ccOHZVB5RUVEKCgrS9OnTLzvmo48+UqtWreTl5aXGjRtr1qxZLqmjNLVcetvYhVucZs6cqeDgYNWvX1/x8fE6c+aMJKlr1676/vvvNWbMGNlsNodu9Jdffql77rlH3t7eCgsL0xNPPOHwfda4cWNNnTpVgwYNkq+vr4YPH67k5GT5+/tr+fLlioiIkI+Pj3r27Knc3FyHOl9//XVFRESoZs2aatGihebPn2+ta9KkiSSpXbt2stls6tq1q9NzPXr0qGJiYtSwYUN5e3vrtttuU1JSkrU+JydHf/jDH+Tv76969eqpb9++OnDgQIlrM23aNIWEhKh58+b6y1/+oo4dO5Y4Vps2bTRlyhSH7S4oLi7WjBkz1KxZM3l5ealRo0aaNm3aNdeBK3OXn9mqyl1/F5GRZCSujnwsX+Qj+Vjp89GgUuvbt6/p2rWrefnll03Dhg3N4cOHTcOGDU1iYqLZs2eP+eqrr0z37t3NfffdZ4wx5tChQ6Z69ermxRdfNNnZ2WbHjh1m3rx55qeffnLxmbi/2NhY07dvX/Pxxx+bmjVrmpycHGOMMYsXLzYXfpS2bt1qPDw8zJQpU8zevXtNUlKS8fb2NklJSRVaR2lrmTRpkmnTpo3DsXx9fc2IESPMnj17zL/+9S9Tq1Yt8/e//90YY8yPP/5oQkNDzZQpU0xubq7Jzc01xhizf/9+U7t2bTN79mzz7bffmvXr15t27dqZwYMHW/sODw83vr6+ZubMmWb//v1m//79JikpydSoUcNERUWZLVu2mIyMDBMREWEGDBhgbffOO++Y4OBg89FHH5nvvvvOfPTRR6ZevXomOTnZGGPM5s2bjSSzcuVKk5uba3788Uen5xofH2/atm1rtmzZYrKzs01qaqr57LPPjDHGnD592kRERJihQ4eaHTt2mN27d5sBAwaY5s2bm6KiIuva+Pj4mIEDB5qvv/7amiSZ/fv3W8e5sGzfvn0OX78Lxo0bZ+rWrWuSk5PN/v37zbp168xrr712zXXg8tzlZ/Zm4E6/i8hIMpKMvDryseKQj+RjZc1HGgWVXH5+vmnQoIHx8PAwixcvNlOnTjU9evRwGJOTk2Mkmb1795qMjAwjyRw4cMBFFVdev/zh7NSpkxk6dKgxxjFcBwwYYLp37+6w3dixY03Lli0rtI7S1uIs5MPDw83Zs2etZQ899JB5+OGHrfnw8HAze/Zsh/3ExcWZ4cOHOyxbt26d8fDwMCdPnrS269evn8OYpKSkEiE5b948ExgYaM03bdrUpKSkOGw3depUY7fbjTHGZGdnG0lm27Ztlz1PY4zp06ePGTJkiNN1//jHP0zz5s1NcXGxtayoqMh4e3ub5cuXG2POX5vAwMASYdumTRszZcoUaz4xMdF07NjRmv/l16+wsNB4eXlZoV6aOnB57vIzezNwp99FZCQZSUZeHflYcchH8rGy5iOPHlRyAQEB+uMf/6iIiAj169dP27dv16pVq+Tj42NNLVq0kCRlZWWpTZs2uv/++xUZGamHHnpIr732Gs+mlcILL7ygt956S3v27HFYvmfPHnXu3NlhWefOnbVv3z6dO3euwuooy1patWqlatWqWfPBwcE6fPjwFbfZvn27kpOTHb4Po6OjVVxcrOzsbGvcnXfeWWLbWrVqqWnTpk6Pd+LECWVlZSkuLs5h388++6yysrKu+ZwkaeTIkXr33XfVtm1bjRs3Ths2bHCof//+/apTp451jHr16unUqVMOx4mMjJSnp6fDfmNiYpSSkiLp/MuM/vnPfyomJsZpDXv27FFRUZHuv/9+p+uvtQ5cnbv8zFZV7vq7iIwkI3F15GP5Ih/Jx8qaj9WvaRTcWvXq1VW9+vkv5fHjx9WnTx+98MILJcYFBwerWrVqSk1N1YYNG7RixQrNnTtXf/3rX7Vp0ybruRxcXZcuXRQdHa3ExEQNHjy4StdRo0YNh3mbzabi4uIrbnP8+HH98Y9/1BNPPFFiXaNGjax/165d+5qOZ4yx9itJr732WonnuH75i+ha9OrVS99//70+//xzpaam6v7771d8fLxmzpyp48ePq0OHDlq0aFGJ7Ro2bHjF+h999FGNHz9eX331lU6ePKmcnBw9/PDDTmvw9va+Yo3XWgeuzl1+Zqsyd/xdREZeREbicsjH8kc+nkc+Vq58pFFQxbRv314fffSRGjdubAXSpWw2mzp37qzOnTtr4sSJCg8P1+LFi5WQkFDB1VZuzz//vNq2bavmzZtbyyIiIrR+/XqHcevXr9ftt99+3SF0I3VUZC2enp4lusvt27fX7t271axZszI7jiQFBgYqJCRE33333WU7rBe6s9fS8W7YsKFiY2MVGxure+65R2PHjtXMmTPVvn17vffeewoICJCvr+911RgaGqp7771XixYt0smTJ9W9e3cFBAQ4HXvbbbfJ29tbaWlpevzxx0usv5E6UJK7/MzeDNzpdxEZWbIeiYyEI/Kx4pCPF5GPjtwtH3n0oIqJj4/XkSNH9Oijj2rLli3KysrS8uXLNWTIEJ07d06bNm3Sc889p61bt+rgwYP6+OOP9cMPPygiIsLVpVc6kZGRiomJ0csvv2wte/LJJ5WWlqapU6fq22+/1VtvvaVXXnlFTz31VIXWUZG1NG7cWGvXrtV///tf642948eP14YNGzRq1ChlZmZq3759+vTTTzVq1KgbPt4zzzyj6dOn6+WXX9a3336rnTt3KikpSS+++KKk87f4eXt7a9myZcrPz1dBQYHT/UycOFGffvqp9u/fr127dmnJkiXWz0FMTIwaNGigvn37at26dcrOztbq1av1xBNPXNPnHsfExOjdd9/VBx98cNlfRtL5jygaP368xo0bp7fffltZWVnauHGj3njjjTKpA47c5Wf2ZuBOv4vISDISV0c+Vhzy8SLysSS3ysdrepMB3NqlLw/59ttvze9+9zvj7+9vvL29TYsWLczo0aNNcXGx2b17t4mOjjYNGzY0Xl5e5vbbbzdz5851XfGVyKVvGjXm/EtPPD09HV4A8+GHH5qWLVuaGjVqmEaNGpm//e1vLqmjNLU4exHNpcf6v//7P3Pvvfda8+np6aZ169bGy8vL4fibN2823bt3Nz4+PqZ27dqmdevWZtq0adZ6Zy+wSUpKMn5+fg7LLn3BjjHGLFq0yLRt29Z4enqaunXrmi5dupiPP/7YWv/aa6+ZsLAw4+Hh8f+1d/dBUV33/8DfS4AFRB4lLFhEBcXFEQiDBnCmPChFUxUttUhMhJrYViKDJtpoEuNDfEoHRWMqMUnlwaKhiUqtjs8BkhI1JgJq3QDyUKyCiQpRJJLqfn5/+NtbrizK16Bi8n7N7Az3nnPPOXd39rOXc885V9XW9t544w3R6/Via2srLi4uEhcXJzU1NUp6Q0ODTJs2Tfr06SNarVYGDhwoM2bMkG+//bbT98akqalJtFqt2NnZdVil+Pbjbt68KcuWLRNvb2/lc1qxYkWX20Gd6ynf2Z+KnvJbxBjJGMkYeXeMjw8W4+MtjI+PVnzUiPz/iRtERERERERE9JPHqQdEREREREREpGBHAREREREREREp2FFARERERERERAp2FBARERERERGRgh0FRERERERERKRgRwERERERERERKdhRQEREREREREQKdhTQj1ZdXR00Gg3KysoAAEVFRdBoNGhubn6o7boXj3LbiahnYBwxT6PRoKCg4GE3g4geIYsXL0ZQUNDDbgbRfcWOAuqRzp49i+nTp8PT0xPW1tbw9vZGWloaLl26dM9lhoeHo6GhAY6Ojt3Y0lsXmaaXpaUl+vXrhxdffBFtbW3dWg8R/TglJycrMcTKygru7u6IiYnBpk2bYDQau62e7o6Bps5Y08vV1RW/+MUvUFpa2i3lmjp5u0tnF/YNDQ0YO3Zst9ZFRA9f+/hk7rV48eL7Wv8333yDmTNnol+/ftBqtdDpdIiNjUVJScl9rbc7sYP5p40dBdTj1NTUICQkBFVVVdi6dSvOnDmDd955B4cOHUJYWBguX758T+VaW1tDp9NBo9F0c4uBrKwsNDQ0oLa2Fhs2bMDmzZuxbNmybq+HiH6cxowZg4aGBtTV1WHPnj2IiopCWloaxo0bhxs3bnRLHfcrBh48eBANDQ3Yt28fWlpaMHbs2E4vKv/73/92a93dQafTQavVPuxmEFE3a2hoUF5r166Fg4ODat/cuXPva/3x8fEoLS1FTk4OKisrsXPnTkRGRv6gm14PUk+M1/RgsaOAepwXXngB1tbW2L9/PyIiItCvXz+MHTsWBw8exLlz5/Dqq68CAPr3748VK1Zg+vTp6N27N/r164d3332303Jv7xXNzs6Gk5MT9u3bB71eD3t7e+Vivb33338fer0eNjY2GDJkCDZs2NChbCcnJ+h0Onh5eWHcuHGIi4vD8ePHlfTq6mrExcXB3d0d9vb2GD58OA4ePKgqo62tDS+//DK8vLyg1Wrh6+uLv/zlL2bPpbW1FWPHjsXIkSPR3NyM5ORkTJw4UZVn9uzZiIyMVLYjIyMxa9YszJo1C46OjujTpw8WLlwIEen0PSOiB8N0t6lv374IDg7GK6+8gr///e/Ys2cPsrOzAQDNzc14/vnn4ebmBgcHB0RHR6O8vBwAUFlZCY1Gg6+++kpVbkZGBnx8fACYvzNUUlKCyMhI2NnZwdnZGbGxsWhqagIAGI1GrFy5EgMGDICtrS0CAwPx0UcfdWi7q6srdDodQkJCkJ6ejgsXLuDo0aPKyID8/HxERETAxsYGeXl5MBqNWLp0KX72s59Bq9UiKCgIe/fuVcobMGAAAOCJJ56ARqNRxbG7xeP//Oc/SExMhIuLC3r16oWQkBAcPXoU2dnZWLJkCcrLy5W7iab39fapBydPnkR0dDRsbW3h6uqK3/3ud2hpaVHSTfE2PT0dHh4ecHV1xQsvvMCLaqIeRqfTKS9HR0doNBrVvg8++OCe4kl7mzdvRv/+/eHo6IgpU6bg6tWrAG7F608//RRvvvkmoqKi4O3tjREjRmDBggWYMGECAPOjp5qbm6HRaFBUVATgf3F79+7dCAgIgI2NDUJDQ3Hq1CnlGNP1bEFBAQYNGgQbGxvExsbi7NmzqrZmZmbCx8cH1tbW8PPzw+bNm1XpGo0GmZmZmDBhAnr16oUZM2YgKioKAODs7AyNRoPk5OR7/jzo0cOOAupRLl++jH379iElJQW2traqNJ1Oh6lTpyI/P1/553b16tUICQlBaWkpUlJSMHPmTFRUVHS5vtbWVqSnp2Pz5s345JNPUF9fr+phzsvLw+uvv47ly5fDYDBgxYoVWLhwIXJycjots7KyEh9//DGefPJJZV9LSwueeuopHDp0CKWlpRgzZgzGjx+P+vp6Jc+0adOwdetWvPXWWzAYDNi4cSPs7e07lN/c3IyYmBgYjUYcOHAATk5OXT7fnJwcWFpa4vPPP8e6deuwZs0avP/++10+nogenOjoaAQGBmL79u0AgMmTJ+Prr7/Gnj178OWXXyI4OBijRo3C5cuXMXjwYISEhCAvL09VRl5eHp5++mmz5ZeVlWHUqFHw9/fH4cOH8c9//hPjx4/HzZs3AQArV65Ebm4u3nnnHfzrX//CnDlz8Mwzz6C4uLjTNpvi9vfff6/smz9/PtLS0mAwGBAbG4t169Zh9erVSE9Px4kTJxAbG4sJEyagqqoKAPD5558D+N9IBdP53y0et7S0ICIiAufOncPOnTtRXl6OP/7xjzAajUhISMBLL72EoUOHKncTExISOrT/2rVriI2NhbOzM44dO4YPP/wQBw8exKxZs1T5CgsLUV1djcLCQuTk5CA7O1vpeCCinu+HxBOT6upqFBQUYNeuXdi1axeKi4uxatUqAIC9vT3s7e1RUFDQLVNR582bh9WrV+PYsWNwc3PD+PHjVZ2Tra2tWL58OXJzc1FSUoLm5mZMmTJFSd+xYwfS0tLw0ksv4dSpU/j973+P3/72tygsLFTVs3jxYkyaNAknT57EkiVLsG3bNgBARUUFGhoasG7duh98LvQIEaIe5MiRIwJAduzYYTZ9zZo1AkAuXLgg3t7e8swzzyhpRqNRHn/8ccnMzBQRkdraWgEgpaWlIiJSWFgoAKSpqUlERLKysgSAnDlzRinjz3/+s7i7uyvbPj4+smXLFlUb3njjDQkLC1O2AYiNjY306tVLtFqtAJBx48bJ999/f8dzHTp0qKxfv15ERCoqKgSAHDhwwGxeU9sNBoMEBARIfHy8tLW1KelJSUkSFxenOiYtLU0iIiKU7YiICNHr9WI0GpV9L7/8suj1+ju2k4juL3PfX5OEhATR6/Xy6aefioODg1y/fl2V7uPjIxs3bhQRkYyMDPHx8VHSTHHFYDCISMcYmJiYKCNHjjRb7/Xr18XOzk4+++wz1f7nnntOEhMTRaRjjG1qapJJkyaJvb29NDY2Kulr165VleHp6SnLly9X7Rs+fLikpKSYLbf9ud4pHm/cuFF69+4tly5dMntOixYtksDAwA772//mvPvuu+Ls7CwtLS1K+u7du8XCwkIaGxtF5Nbn5e3tLTdu3FDyTJ48WRISEszWS0QPX1ZWljg6Oirb3RFP7Ozs5MqVK8q+efPmyZNPPqlsf/TRR+Ls7Cw2NjYSHh4uCxYskPLyciXdXKxramoSAFJYWCgi/4vbH3zwgZLn0qVLYmtrK/n5+cq5AZAjR44oeQwGgwCQo0ePiohIeHi4zJgxQ3UOkydPlqeeekrZBiCzZ89W5bn9d4N+WjiigHok6eJw+ICAAOVv05Cyr7/+usv12NnZKcNyAcDDw0M5/tq1a6iursZzzz2n9Azb29tj2bJlqK6uVpWTkZGBsrIylJeXY9euXaisrMSzzz6rpLe0tGDu3LnQ6/VwcnKCvb09DAaDMqKgrKwMjz32GCIiIu7Y3piYGPj6+iI/Px/W1tZdPk+T0NBQ1fzksLAwVFVVKXcQiahnERFoNBqUl5ejpaUFrq6uqnhUW1urxKMpU6agrq4OR44cAXDrjllwcDCGDBlitmzTiAJzzpw5g9bWVsTExKjqy83N7RD/wsPDYW9vD2dnZ5SXlyM/Px/u7u5KekhIiPL3lStXcP78eYwcOVJVxsiRI2EwGDp9H7oSj8vKyvDEE0/AxcWl03LuxmAwIDAwEL169VK1zWg0qkarDR06FI899piy3f63g4h6tu6KJ/3790fv3r2V7dvjQHx8PM6fP4+dO3dizJgxKCoqQnBw8D2NPgoLC1P+dnFxgZ+fnypmWlpaYvjw4cr2kCFD4OTkpOQxGAxdirvt4zWR5cNuAFF7vr6+0Gg0MBgMmDRpUod0g8EAZ2dnuLm5AQCsrKxU6RqN5v+0Sri5402dFKY5qe+9955qGgEA1QUicGtahK+vLwDAz88PV69eRWJiIpYtWwZfX1/MnTsXBw4cQHp6Onx9fWFra4tf//rXyvDc26dZdOaXv/wltm3bhtOnT2PYsGHKfgsLiw6dK5wvS/ToMxgMGDBgAFpaWuDh4aHMW23PNP1Ip9MhOjoaW7ZsQWhoKLZs2YKZM2d2Wvad4o4p/u3evRt9+/ZVpd2+8F9+fj78/f3h6upqdipU+3+671VX4nFX42h3+KG/PUT08HRXPOlKHLCxsUFMTAxiYmKwcOFCPP/881i0aBGSk5NhYXHrfm3767eHfe3WHfGafjw4ooB6FFdXV8TExGDDhg347rvvVGmNjY3Iy8tDQkLCfXlywe3c3d3h6emJmpoa+Pr6ql6mxbY6Y/qhMZ1DSUkJkpOTMWnSJAwbNgw6nQ51dXVK/mHDhsFoNN5x7i8ArFq1CklJSRg1ahROnz6t7Hdzc+uwCKO5R4vdvgjPkSNHMGjQoA4dH0T08H388cc4efIk4uPjERwcjMbGRlhaWnaIR3369FGOMa3jcvjwYdTU1KjmqN4uICAAhw4dMpvm7+8PrVaL+vr6DvV5eXmp8np5ecHHx6dL66U4ODjA09Ozw+PBSkpK4O/vDwDKaKn2I526Eo8DAgJQVlbW6ZNxrK2t7zp6Sq/Xo7y8HNeuXVO1zcLCAn5+fnc9PyLq+bojntwrf39/Jb6Ybnq1v37r7LGwppFiANDU1ITKykro9Xpl340bN/DFF18o2xUVFWhublby6PX6O8bdzpiLx/TTwY4C6nHefvtttLW1ITY2Fp988gnOnj2LvXv3IiYmBn379sXy5csfWFuWLFmClStX4q233kJlZSVOnjyJrKwsrFmzRpWvubkZjY2NOH/+PIqLi7F06VIMHjxYCdCDBg3C9u3blekJTz/9tKrXuX///khKSsL06dNRUFCA2tpaFBUV4W9/+1uHNqWnp2Pq1KmIjo5WVjiPjo7GF198gdzcXFRVVWHRokWqFXFN6uvr8eKLL6KiogJbt27F+vXrkZaW1p1vGRHdg7a2NjQ2NuLcuXM4fvw4VqxYgbi4OIwbNw7Tpk3D6NGjERYWhokTJ2L//v2oq6vDZ599hldffVV1cfirX/0KV69excyZMxEVFQVPT89O61ywYAGOHTuGlJQUnDhxAl999RUyMzNx8eJF9O7dG3PnzsWcOXOQk5OD6upqHD9+HOvXr7/jYq5dMW/ePLz55pvIz89HRUUF5s+fj7KyMiUWPf7447C1tcXevXtx4cIFfPvttwDuHo8TExOh0+kwceJElJSUoKamBtu2bcPhw4cB3IqztbW1KCsrw8WLF80uMDZ16lTY2NggKSkJp06dQmFhIVJTU/Hss8+qplMQ0aPth8aTu7l06RKio6Px17/+FSdOnEBtbS0+/PBD/OlPf0JcXByAW6MWQkNDsWrVKhgMBhQXF+O1114zW97SpUtx6NAhnDp1CsnJyejTp4/qaVdWVlZITU3F0aNH8eWXXyI5ORmhoaEYMWIEgFtxNzs7G5mZmaiqqsKaNWuwffv2uz4i0tvbGxqNBrt27cI333yjegIM/QQ81BUSiDpRV1cnSUlJ4u7uLlZWVuLl5SWpqaly8eJFJY+3t7dkZGSojgsMDJRFixaJSNcWM2y/sI2IyI4dO+T2r0VeXp4EBQWJtbW1ODs7y89//nPZvn27kg5AeWk0GvHw8JCEhASprq5W8tTW1kpUVJTY2tqKl5eXvP322xIRESFpaWlKnu+++07mzJkjHh4eYm1tLb6+vrJp0yazbRcRSU1NFQ8PD6moqBARkddff13c3d3F0dFR5syZI7NmzeqwmGFKSor84Q9/EAcHB3F2dpZXXnlFtbghET14SUlJSgyxtLQUNzc3GT16tGzatElu3ryp5Lty5YqkpqaKp6enEhenTp0q9fX1qvJ+85vfCAAlfpiYiyNFRUUSHh4uWq1WnJycJDY2Vkk3Go2ydu1a8fPzEysrK3Fzc5PY2FgpLi4Wkc4XHTTpLP3mzZuyePFi6du3r1hZWUlgYKDs2bNHlee9994TLy8vsbCwUMWxu8Xjuro6iY+PFwcHB7Gzs5OQkBBlMa/r169LfHy8ODk5CQDJysoSEfVihiIiJ06ckKioKLGxsREXFxeZMWOGXL16VfV53W3xWCLqWcxd8/2QeGJucdSMjAzx9vYWkVvxZv78+RIcHCyOjo5iZ2cnfn5+8tprr0lra6tyzOnTpyUsLExsbW0lKChI9u/fb3Yxw3/84x8ydOhQsba2lhEjRqgWRTSd27Zt22TgwIGi1Wpl9OjR8u9//1vVvg0bNsjAgQPFyspKBg8eLLm5uar022OhydKlS0Wn04lGo5GkpKQuvNv0Y6ER4UPUiX4KIiMjERQUhLVr1z7sphARERHRXRQVFSEqKgpNTU2dTu/Kzs7G7Nmz0dzc/EDbRj9+nHpARERERERERAp2FBARERERERGRglMPiIiIiIiIiEjBEQVEREREREREpGBHAREREREREREp2FFARERERERERAp2FBARERERERGRgh0FRERERERERKRgRwERERERERERKdhRQEREREREREQKdhQQERERERERkYIdBURERERERESk+H/U79LjMIOmCQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1200x700 with 6 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)\n",
    "\n",
    "sns.countplot(x =\"StreamingTV\", data=df, ax=axes[0,0])\n",
    "sns.countplot(x =\"StreamingMovies\", data=df, ax=axes[0,1])\n",
    "sns.countplot(x =\"OnlineSecurity\", data=df, ax=axes[0,2])\n",
    "sns.countplot(x =\"OnlineBackup\", data=df, ax=axes[1,0])\n",
    "sns.countplot(x =\"DeviceProtection\", data=df, ax=axes[1,1])\n",
    "sns.countplot(x =\"TechSupport\", data=df, ax=axes[1,2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>StreamingTV</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.335231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.300702</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "StreamingTV                  \n",
       "No                   0.335231\n",
       "No internet service  0.074050\n",
       "Yes                  0.300702"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['StreamingTV','Churn']].groupby('StreamingTV').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.336804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.299414</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "StreamingMovies              \n",
       "No                   0.336804\n",
       "No internet service  0.074050\n",
       "Yes                  0.299414"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['StreamingMovies','Churn']].groupby('StreamingMovies').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.417667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.146112</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "OnlineSecurity               \n",
       "No                   0.417667\n",
       "No internet service  0.074050\n",
       "Yes                  0.146112"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['OnlineSecurity','Churn']].groupby('OnlineSecurity').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>OnlineBackup</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.399288</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.215315</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "OnlineBackup                 \n",
       "No                   0.399288\n",
       "No internet service  0.074050\n",
       "Yes                  0.215315"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['OnlineBackup','Churn']].groupby('OnlineBackup').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.391276</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.225021</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "DeviceProtection             \n",
       "No                   0.391276\n",
       "No internet service  0.074050\n",
       "Yes                  0.225021"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['DeviceProtection','Churn']].groupby('DeviceProtection').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TechSupport</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.416355</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No internet service</th>\n",
       "      <td>0.074050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.151663</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        Churn\n",
       "TechSupport                  \n",
       "No                   0.416355\n",
       "No internet service  0.074050\n",
       "Yes                  0.151663"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['TechSupport','Churn']].groupby('TechSupport').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "All internet service related features seem to have different churn rates for their classes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Phone service"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PhoneService\n",
       "Yes    6361\n",
       "No      682\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.PhoneService.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='PhoneService', ylabel='count'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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y0OArz74wIED+uMf/6isrCy1bNnSV23UyW63y263+7oNAADQRHx2hig3N1dlZWXq06ePAgICFBAQoOzsbM2fP18BAQEKDQ1VVVWVysvLvd5XWlqqsLAwSVJYWNh53zqrff1zNQ6HQ0FBQY20OwAAcDnxWSAaPHiw8vPzlZeXZx033XST4uPjrT+3aNFC69evt95TWFio4uJiuVwuSZLL5VJ+fr7KysqsmqysLDkcDkVFRVk1565RW1O7BgAAgM8umbVt21Y33HCD11jr1q3Vvn17a3zs2LFKTExUu3bt5HA49OSTT8rlcql///6SpCFDhigqKkoPP/ywZs+erZKSEj377LNKSEiwLnk98cQTWrBggaZPn64xY8Zow4YNWrFihdLT05t2wwAAoNnyWSC6EHPnzpWfn59GjBihyspKxcbGauHChda8v7+/1qxZowkTJsjlcql169YaPXq0UlNTrZouXbooPT1dU6ZM0bx58xQREaF33nlHsbGxvtgSAABohnz+HKLLwcU8xwAA6sJziIC68RwiAACAZoJABAAAjEcgAgAAxiMQAQAA4xGIAACA8QhEAADAeAQiAABgPAIRAAAwHoEIAAAYj0AEAACMRyACAADGIxABAADj1SsQ3XbbbSovLz9v3O1267bbbrvUngAAAJpUvQLRxo0bVVVVdd746dOntXnz5ktuCgAAoCkFXEzxzp07rT/v3r1bJSUl1uvq6mplZGTol7/8ZcN1BwAA0AQuKhD17t1bNptNNputzktjQUFBev311xusOQAAgKZwUYGoqKhIHo9Hv/rVr7Rjxw516NDBmgsMDFRISIj8/f0bvEkAAIDGdFGBqHPnzpKkmpqaRmkGAADAFy4qEJ1r3759+uKLL1RWVnZeQEpJSbnkxgAAAJpKvQLR22+/rQkTJuiqq65SWFiYbDabNWez2QhEAADgslKvQDRz5ky9+OKLSkpKauh+AAAAmly9nkN07Ngx3XfffQ3dCwAAgE/UKxDdd999WrduXUP3AgAA4BP1umTWtWtXPffcc9q2bZuio6PVokULr/lJkyY1SHMAAABNwebxeDwX+6YuXbr86wVtNn3//feX1FRz43a75XQ6VVFRIYfD4et2AFyGilOjfd0C0Cx1SslvtLUv5ud3vc4QFRUV1asxAACA5qhe9xABAABcSep1hmjMmDE/Of/ee+/V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axUVFaXXX39d3bp1U1FRkY93gaZkzC93BX7Kyy+/rN69e6tbt27WWI8ePbRlyxavui1btui6666Tv79/U7cIoJ769Omjjz/+WFdffbUVkv6ZzWbTgAEDNGDAAKWkpKhz585atWqVEhMTm7hb+ApniABJ0dHRio+P1/z5862xp556SuvXr9ef/vQn/eUvf9GSJUu0YMECTZ061YedArhYCQkJOnr0qEaNGqWvvvpK+/fvV2Zmph577DFVV1dr+/bteumll/T111+ruLhYn3zyiY4cOaIePXr4unU0IQIR8P+kpqaqpqbGet2nTx+tWLFCH374oW644QalpKQoNTVVjz76qO+aBHDRwsPDtWXLFlVXV2vIkCGKjo7W5MmTFRwcLD8/PzkcDm3atEl33nmnrrvuOj377LOaM2eO7rjjDl+3jibEt8wAAIDxOEMEAACMRyACAADGIxABAADjEYgAAIDxCEQAAMB4BCIAAGA8AhEAADAegQgAABiPQASgyaSlpSk4ONjXbfjco48+qnvuucfXbQA4B4EIQIN69NFHZbPZZLPZFBgYqK5duyo1NVVnz571dWteioqK9OCDDyo8PFwtW7ZURESEhg0bpr179zb6Z8+bN09paWmN/jkALhy/7R5Agxs6dKjef/99VVZW6r//+7+VkJCgFi1aqGPHjr5uTZJ05swZ3X777erWrZs++eQTdezYUQcPHtTatWtVXl5e73WrqqoUGBj4s3VOp7PenwGgcXCGCECDs9vtCgsLU+fOnTVhwgTFxMTos88+s+YzMzPVo0cPtWnTRkOHDtXhw4etuZqaGqWmpioiIkJ2u129e/dWRkaGNf/DDz/IZrPpk08+0aBBg9SqVSv16tVLOTk5Xj18+eWXuvXWWxUUFKTIyEhNmjRJJ0+elCQVFBRo//79Wrhwofr376/OnTtrwIABmjlzpvr372+tceDAAd1///0KDg5Wu3btNGzYMP3www/WfO2lrxdffFHh4eHq1q2b/v3f/139+vU77++kV69eSk1N9XrfuXuePXu2unbtKrvdrk6dOunFF1+84D4AXDoCEYBGFxQUpKqqKknSP/7xD/35z3/WBx98oE2bNqm4uFhTp061aufNm6c5c+boz3/+s3bu3KnY2Fj927/9m/bt2+e15jPPPKOpU6cqLy9P1113nUaNGmVdltu/f7+GDh2qESNGaOfOnVq+fLm+/PJLTZw4UZLUoUMH+fn56aOPPlJ1dXWdPZ85c0axsbFq27atNm/erC1btlgBrnYvkrR+/XoVFhYqKytLa9asUXx8vHbs2KH9+/dbNQUFBdq5c6cefPDBOj8rOTlZL7/8sp577jnt3r1by5YtU2ho6EX1AeASeQCgAY0ePdozbNgwj8fj8dTU1HiysrI8drvdM3XqVM/777/vkeT57rvvrPo33njDExoaar0ODw/3vPjii15r3nzzzZ4//OEPHo/H4ykqKvJI8rzzzjvWfEFBgUeSZ8+ePR6Px+MZO3asZ/z48V5rbN682ePn5+c5deqUx+PxeBYsWOBp1aqVp23btp5BgwZ5UlNTPfv377fqP/jgA0+3bt08NTU11lhlZaUnKCjIk5mZae01NDTUU1lZ6fVZvXr18qSmplqvk5OTPf369avz78jtdnvsdrvn7bffrvPv80L6AHDpOEMEoMGtWbNGbdq0UcuWLXXHHXfogQce0IwZMyRJrVq10jXXXGPVduzYUWVlZZIkt9utQ4cOacCAAV7rDRgwQHv27PEa69mzp9cakqx1vv32W6WlpalNmzbWERsbq5qaGhUVFUmSEhISVFJSoqVLl8rlcmnlypW6/vrrlZWVZa3x3XffqW3bttYa7dq10+nTp73O/kRHR59331B8fLyWLVsmSfJ4PPqv//ovxcfH1/l3tWfPHlVWVmrw4MF1zl9oHwAuDTdVA2hwgwYN0qJFixQYGKjw8HAFBPz//9W0aNHCq9Zms8nj8Vz0Z5y7js1mk/TjvTiSdOLECf3+97/XpEmTzntfp06drD+3bdtWd999t+6++27NnDlTsbGxmjlzpm6//XadOHFCffv21dKlS89bo0OHDtafW7dufd78qFGjlJSUpG+++UanTp3SgQMH9MADD9S5j6CgoJ/c54X2AeDSEIgANLjWrVura9euF/0+h8Oh8PBwbdmyRb/5zW+s8S1btujXv/71Ba/Tp08f7d69+6J6sNls6t69u7Zu3WqtsXz5coWEhMjhcFz4JiRFREToN7/5jZYuXapTp07p9ttvV0hISJ211157rYKCgrR+/Xo9/vjjde6lvn0AuHBcMgPQrEybNk2vvPKKli9frsLCQj399NPKy8vTH//4xwteIykpSVu3btXEiROVl5enffv26dNPP7Vuqs7Ly9OwYcP00Ucfaffu3fruu+/07rvv6r333tOwYcMk/XjZ66qrrtKwYcO0efNmFRUVaePGjZo0aZIOHjz4sz3Ex8frww8/1MqVK//l5TJJatmypZKSkjR9+nT953/+p/bv369t27bp3XffbZA+AFwYzhABaFYmTZqkiooKPfXUUyorK1NUVJQ+++wzXXvttRe8Rs+ePZWdna1nnnlGt956qzwej6655hrrslVERISuvvpqvfDCC9bX+GtfT5kyRdKP9zpt2rRJSUlJGj58uI4fP65f/vKXGjx48AWdqbn33ns1ceJE+fv7/+xTqZ977jkFBAQoJSVFhw4dUseOHfXEE080SB8ALozNU5+L9wAAAFcQLpkBAADjEYgAAIDxCEQAAMB4BCIAAGA8AhEAADAegQgAABiPQAQAAIxHIAIAAMYjEAEAAOMRiAAAgPEIRAAAwHj/H++ydEqpukyLAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = 'PhoneService', data = df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultipleLines\n",
       "No                  3390\n",
       "Yes                 2971\n",
       "No phone service     682\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.MultipleLines.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='MultipleLines', ylabel='count'>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAGwCAYAAABIC3rIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA1tklEQVR4nO3df3RNd77/8ddJSCQ4Jw1JTlLxY2hJNH625VzDKGmC6HCrP7TqRynDRDukQ27WGDX0jlarqCptXWLuyNBpMVPGj5SKDkFlmlIlg4mJLpKYIgclSPb3j7nZ3x5BNZKcw34+1tprZe/9Pp/93pqTvLr35+zYDMMwBAAAYGF+3m4AAADA2whEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8up4u4HbQXl5uY4fP66GDRvKZrN5ux0AAHATDMPQ2bNnFRUVJT+/G18DIhDdhOPHjys6OtrbbQAAgCo4duyYmjRpcsMaAtFNaNiwoaR//4Pa7XYvdwMAAG6G2+1WdHS0+Xv8RghEN6HiNpndbicQAQBwm7mZ6S5enVS9cOFCtWvXzgwaLpdL69evN/f37NlTNpvNYxk7dqzHGAUFBUpKSlJwcLDCw8M1adIkXblyxaNm69at6tSpkwIDA9WqVSulp6fXxukBAIDbhFevEDVp0kSvvPKK7rnnHhmGoWXLlmnAgAH6/PPP1bZtW0nS6NGjNX36dPM1wcHB5tdlZWVKSkqS0+nUjh07dOLECQ0bNkx169bVb3/7W0lSfn6+kpKSNHbsWC1fvlybN2/Wc889p8jISCUmJtbuCQMAAJ9k87W/dh8aGqrXXntNo0aNUs+ePdWhQwfNnTv3mrXr169X//79dfz4cUVEREiSFi1apNTUVJ08eVIBAQFKTU3VunXr9OWXX5qvGzx4sM6cOaMNGzZcc9zS0lKVlpaa6xX3IEtKSrhlBgDAbcLtdsvhcNzU72+feQ5RWVmZVqxYofPnz8vlcpnbly9frsaNG+u+++5TWlqavv32W3Nfdna24uLizDAkSYmJiXK73dq/f79ZEx8f73GsxMREZWdnX7eXmTNnyuFwmAufMAMA4M7m9UnV+/btk8vl0sWLF9WgQQOtXr1asbGxkqSnn35azZo1U1RUlPbu3avU1FTl5eVp1apVkqTCwkKPMCTJXC8sLLxhjdvt1oULFxQUFFSpp7S0NKWkpJjrFVeIAADAncnrgah169bKzc1VSUmJPvjgAw0fPlxZWVmKjY3VmDFjzLq4uDhFRkaqd+/eOnLkiFq2bFljPQUGBiowMLDGxgcAAL7F67fMAgIC1KpVK3Xu3FkzZ85U+/btNW/evGvWdunSRZJ0+PBhSZLT6VRRUZFHTcW60+m8YY3dbr/m1SEAAGA9Xg9EVysvL/eY0Pxdubm5kqTIyEhJksvl0r59+1RcXGzWZGZmym63m7fdXC6XNm/e7DFOZmamxzwlAABgbV69ZZaWlqa+ffuqadOmOnv2rDIyMrR161Zt3LhRR44cUUZGhvr166dGjRpp7969mjhxonr06KF27dpJkhISEhQbG6uhQ4dq1qxZKiws1JQpU5ScnGze8ho7dqzeeustTZ48WSNHjtSWLVv0/vvva926dd48dQAA4EO8GoiKi4s1bNgwnThxQg6HQ+3atdPGjRv18MMP69ixY/r44481d+5cnT9/XtHR0Ro0aJCmTJlivt7f319r167VuHHj5HK5VL9+fQ0fPtzjuUUtWrTQunXrNHHiRM2bN09NmjTR4sWLeQYRAAAw+dxziHzRD3mOAQAA8A235XOIAAAAvIVABAAALI9ABAAALI9ABAAALM/rT6oG4D0F0+O83QJ8TNOp+7zdAuAVXCECAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACW59VAtHDhQrVr1052u112u10ul0vr168391+8eFHJyclq1KiRGjRooEGDBqmoqMhjjIKCAiUlJSk4OFjh4eGaNGmSrly54lGzdetWderUSYGBgWrVqpXS09Nr4/QAAMBtwquBqEmTJnrllVeUk5OjPXv2qFevXhowYID2798vSZo4caI++ugj/fGPf1RWVpaOHz+uRx991Hx9WVmZkpKSdOnSJe3YsUPLli1Tenq6pk6datbk5+crKSlJDz30kHJzczVhwgQ999xz2rhxY62fLwAA8E02wzAMbzfxXaGhoXrttdf02GOPKSwsTBkZGXrsscckSQcPHlRMTIyys7PVtWtXrV+/Xv3799fx48cVEREhSVq0aJFSU1N18uRJBQQEKDU1VevWrdOXX35pHmPw4ME6c+aMNmzYcM0eSktLVVpaaq673W5FR0erpKREdru9Bs8eqF0F0+O83QJ8TNOp+7zdAlBt3G63HA7HTf3+9pk5RGVlZVqxYoXOnz8vl8ulnJwcXb58WfHx8WZNmzZt1LRpU2VnZ0uSsrOzFRcXZ4YhSUpMTJTb7TavMmVnZ3uMUVFTMca1zJw5Uw6Hw1yio6Or81QBAICP8Xog2rdvnxo0aKDAwECNHTtWq1evVmxsrAoLCxUQEKCQkBCP+oiICBUWFkqSCgsLPcJQxf6KfTeqcbvdunDhwjV7SktLU0lJibkcO3asOk4VAAD4qDrebqB169bKzc1VSUmJPvjgAw0fPlxZWVle7SkwMFCBgYFe7QEAANQerweigIAAtWrVSpLUuXNnffbZZ5o3b56efPJJXbp0SWfOnPG4SlRUVCSn0ylJcjqd2r17t8d4FZ9C+27N1Z9MKyoqkt1uV1BQUE2dFgCgirrN7+btFuBDtj+/vVaO4/VbZlcrLy9XaWmpOnfurLp162rz5s3mvry8PBUUFMjlckmSXC6X9u3bp+LiYrMmMzNTdrtdsbGxZs13x6ioqRgDAADAq1eI0tLS1LdvXzVt2lRnz55VRkaGtm7dqo0bN8rhcGjUqFFKSUlRaGio7Ha7nn/+eblcLnXt2lWSlJCQoNjYWA0dOlSzZs1SYWGhpkyZouTkZPOW19ixY/XWW29p8uTJGjlypLZs2aL3339f69at8+apAwAAH+LVQFRcXKxhw4bpxIkTcjgcateunTZu3KiHH35YkjRnzhz5+flp0KBBKi0tVWJiot5++23z9f7+/lq7dq3GjRsnl8ul+vXra/jw4Zo+fbpZ06JFC61bt04TJ07UvHnz1KRJEy1evFiJiYm1fr4AAMA3+dxziHzRD3mOAXA74TlEuJovPIeIOUT4rluZQ3RbPocIAADAWwhEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8rwaiGbOnKkHHnhADRs2VHh4uAYOHKi8vDyPmp49e8pms3ksY8eO9agpKChQUlKSgoODFR4erkmTJunKlSseNVu3blWnTp0UGBioVq1aKT09vaZPDwAA3Ca8GoiysrKUnJysnTt3KjMzU5cvX1ZCQoLOnz/vUTd69GidOHHCXGbNmmXuKysrU1JSki5duqQdO3Zo2bJlSk9P19SpU82a/Px8JSUl6aGHHlJubq4mTJig5557Ths3bqy1cwUAAL6rjjcPvmHDBo/19PR0hYeHKycnRz169DC3BwcHy+l0XnOMTZs26auvvtLHH3+siIgIdejQQTNmzFBqaqqmTZumgIAALVq0SC1atNDs2bMlSTExMfrrX/+qOXPmKDExsdKYpaWlKi0tNdfdbnd1nC4AAPBRPjWHqKSkRJIUGhrqsX358uVq3Lix7rvvPqWlpenbb78192VnZysuLk4RERHmtsTERLndbu3fv9+siY+P9xgzMTFR2dnZ1+xj5syZcjgc5hIdHV0t5wcAAHyTV68QfVd5ebkmTJigbt266b777jO3P/3002rWrJmioqK0d+9epaamKi8vT6tWrZIkFRYWeoQhSeZ6YWHhDWvcbrcuXLigoKAgj31paWlKSUkx191uN6EIAIA7mM8EouTkZH355Zf661//6rF9zJgx5tdxcXGKjIxU7969deTIEbVs2bJGegkMDFRgYGCNjA0AAHyPT9wyGz9+vNauXatPPvlETZo0uWFtly5dJEmHDx+WJDmdThUVFXnUVKxXzDu6Xo3dbq90dQgAAFiPVwORYRgaP368Vq9erS1btqhFixbf+5rc3FxJUmRkpCTJ5XJp3759Ki4uNmsyMzNlt9sVGxtr1mzevNljnMzMTLlcrmo6EwAAcDvzaiBKTk7W73//e2VkZKhhw4YqLCxUYWGhLly4IEk6cuSIZsyYoZycHB09elR//vOfNWzYMPXo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AFgegQgAAFgegQgAAFgegQgAAFgegQgAAFgegQgAAFgegQgAAFgegQgAAFieVwPRtm3b9MgjjygqKko2m01r1qzx2G8YhqZOnarIyEgFBQUpPj5ehw4d8qg5deqUhgwZIrvdrpCQEI0aNUrnzp3zqNm7d6+6d++uevXqKTo6WrNmzarpUwMAALcRrwai8+fPq3379lqwYME198+aNUtvvvmmFi1apF27dql+/fpKTEzUxYsXzZohQ4Zo//79yszM1Nq1a7Vt2zaNGTPG3O92u5WQkKBmzZopJydHr732mqZNm6Z33323xs8PAADcHup48+B9+/ZV3759r7nPMAzNnTtXU6ZM0YABAyRJv/vd7xQREaE1a9Zo8ODBOnDggDZs2KDPPvtM999/vyRp/vz56tevn15//XVFRUVp+fLlunTpkpYsWaKAgAC1bdtWubm5euONNzyCEwAAsC6fnUOUn5+vwsJCxcfHm9scDoe6dOmi7OxsSVJ2drZCQkLMMCRJ8fHx8vPz065du8yaHj16KCAgwKxJTExUXl6eTp8+fc1jl5aWyu12eywAAODO5bOBqLCwUJIUERHhsT0iIsLcV1hYqPDwcI/9derUUWhoqEfNtcb47jGuNnPmTDkcDnOJjo6+9RMCAAA+y2cDkTelpaWppKTEXI4dO+btlgAAQA3y2UDkdDolSUVFRR7bi4qKzH1Op1PFxcUe+69cuaJTp0551FxrjO8e42qBgYGy2+0eCwAAuHP5bCBq0aKFnE6nNm/ebG5zu93atWuXXC6XJMnlcunMmTPKyckxa7Zs2aLy8nJ16dLFrNm2bZsuX75s1mRmZqp169a66667aulsAACAL/NqIDp37pxyc3OVm5sr6d8TqXNzc1VQUCCbzaYJEybo5Zdf1p///Gft27dPw4YNU1RUlAYOHChJiomJUZ8+fTR69Gjt3r1b27dv1/jx4zV48GBFRUVJkp5++mkFBARo1KhR2r9/v1auXKl58+YpJSXFS2cNAAB8jVc/dr9nzx499NBD5npFSBk+fLjS09M1efJknT9/XmPGjNGZM2f04x//WBs2bFC9evXM1yxfvlzjx49X79695efnp0GDBunNN9809zscDm3atEnJycnq3LmzGjdurKlTp/KRewAAYLIZhmF4uwlf53a75XA4VFJSwnwi3FEKpsd5uwX4mKZT93m7BXWb383bLcCHbH9+e5Vf+0N+f/vsHCIAAIDaQiACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWV8fbDVhJ50m/83YL8CE5rw3zdgsAgP/DFSIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5BCIAAGB5Ph2Ipk2bJpvN5rG0adPG3H/x4kUlJyerUaNGatCggQYNGqSioiKPMQoKCpSUlKTg4GCFh4dr0qRJunLlSm2fCgAA8GF1vN3A92nbtq0+/vhjc71Onf/f8sSJE7Vu3Tr98Y9/lMPh0Pjx4/Xoo49q+/btkqSysjIlJSXJ6XRqx44dOnHihIYNG6a6devqt7/9ba2fCwAA8E0+H4jq1Kkjp9NZaXtJSYn+53/+RxkZGerVq5ckaenSpYqJidHOnTvVtWtXbdq0SV999ZU+/vhjRUREqEOHDpoxY4ZSU1M1bdo0BQQE1PbpAAAAH+TTt8wk6dChQ4qKitKPfvQjDRkyRAUFBZKknJwcXb58WfHx8WZtmzZt1LRpU2VnZ0uSsrOzFRcXp4iICLMmMTFRbrdb+/fvv+4xS0tL5Xa7PRYAAHDn8ulA1KVLF6Wnp2vDhg1auHCh8vPz1b17d509e1aFhYUKCAhQSEiIx2siIiJUWFgoSSosLPQIQxX7K/Zdz8yZM+VwOMwlOjq6ek8MAAD4FJ++Zda3b1/z63bt2qlLly5q1qyZ3n//fQUFBdXYcdPS0pSSkmKuu91uQhEAAHcwn75CdLWQkBDde++9Onz4sJxOpy5duqQzZ8541BQVFZlzjpxOZ6VPnVWsX2teUoXAwEDZ7XaPBQAA3Lluq0B07tw5HTlyRJGRkercubPq1q2rzZs3m/vz8vJUUFAgl8slSXK5XNq3b5+Ki4vNmszMTNntdsXGxtZ6/wAAwDf59C2zX/7yl3rkkUfUrFkzHT9+XC+99JL8/f311FNPyeFwaNSoUUpJSVFoaKjsdruef/55uVwude3aVZKUkJCg2NhYDR06VLNmzVJhYaGmTJmi5ORkBQYGevnsAACAr/DpQPT111/rqaee0jfffKOwsDD9+Mc/1s6dOxUWFiZJmjNnjvz8/DRo0CCVlpYqMTFRb7/9tvl6f39/rV27VuPGjZPL5VL9+vU1fPhwTZ8+3VunBAAAfJBPB6IVK1bccH+9evW0YMECLViw4Lo1zZo101/+8pfqbg0AANxBbqs5RAAAADWBQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACzPUoFowYIFat68uerVq6cuXbpo9+7d3m4JAAD4AMsEopUrVyolJUUvvfSS/va3v6l9+/ZKTExUcXGxt1sDAABeZplA9MYbb2j06NF69tlnFRsbq0WLFik4OFhLlizxdmsAAMDL6ni7gdpw6dIl5eTkKC0tzdzm5+en+Ph4ZWdnV6ovLS1VaWmpuV5SUiJJcrvdt9RHWemFW3o97iy3+v1UHc5eLPN2C/AxvvB9eeXCFW+3AB9yK9+TFa81DON7ay0RiP71r3+prKxMERERHtsjIiJ08ODBSvUzZ87Ub37zm0rbo6Oja6xHWI9j/lhvtwBUNtPh7Q4AD47UW/+ePHv2rByOG49jiUD0Q6WlpSklJcVcLy8v16lTp9SoUSPZbDYvdnb7c7vdio6O1rFjx2S3273dDsD3JHwS35fVwzAMnT17VlFRUd9ba4lA1LhxY/n7+6uoqMhje1FRkZxOZ6X6wMBABQYGemwLCQmpyRYtx2638yaHT+F7Er6I78tb931XhipYYlJ1QECAOnfurM2bN5vbysvLtXnzZrlcLi92BgAAfIElrhBJUkpKioYPH677779fDz74oObOnavz58/r2Wef9XZrAADAyywTiJ588kmdPHlSU6dOVWFhoTp06KANGzZUmmiNmhUYGKiXXnqp0i1JwFv4noQv4vuy9tmMm/ksGgAAwB3MEnOIAAAAboRABAAALI9ABAAALI9ABEnSiBEjNHDgQG+34TOOHj0qm82m3Nxcb7cCAKgFBCIfMWLECNlsNr3yyise29esWcPTsb0gOjpaJ06c0H333eftVlANeH/BVxiGofj4eCUmJlba9/bbbyskJERff/21FzoDgciH1KtXT6+++qpOnz7t7VbuaJcvX/7eGn9/fzmdTtWpY5knU9zxeH/BF9hsNi1dulS7du3SO++8Y27Pz8/X5MmTNX/+fDVp0sSLHVoXgciHxMfHy+l0aubMmTes+/DDD9W2bVsFBgaqefPmmj179g3rp02bpg4dOuidd95RdHS0goOD9cQTT6ikpKRS7euvv67IyEg1atRIycnJHuHh9OnTGjZsmO666y4FBwerb9++OnTokLk/PT1dISEh2rhxo2JiYtSgQQP16dNHJ06c8DjG4sWLFRMTo3r16qlNmzZ6++23b9j/Bx98oLi4OAUFBalRo0aKj4/X+fPnb2q8iltfK1eu1E9+8hPVq1dPCxcuVFBQkNavX+9xnNWrV6thw4b69ttvr3nLbP/+/erfv7/sdrsaNmyo7t2768iRI1U+L9Sum3l//dD3FlAV0dHRmjdvnn75y18qPz9fhmFo1KhRSkhIUMeOHdW3b181aNBAERERGjp0qP71r3+Zr/2+n4e4BQZ8wvDhw40BAwYYq1atMurVq2ccO3bMMAzDWL16tfHd/0x79uwx/Pz8jOnTpxt5eXnG0qVLjaCgIGPp0qXXHfull14y6tevb/Tq1cv4/PPPjaysLKNVq1bG008/7XF8u91ujB071jhw4IDx0UcfGcHBwca7775r1vz0pz81YmJijG3bthm5ublGYmKi0apVK+PSpUuGYRjG0qVLjbp16xrx8fHGZ599ZuTk5BgxMTEex/n9739vREZGGh9++KHxj3/8w/jwww+N0NBQIz09/Zq9Hz9+3KhTp47xxhtvGPn5+cbevXuNBQsWGGfPnr2p8fLz8w1JRvPmzc2a48ePG4899pjxzDPPeBxr0KBB5raK133++eeGYRjG119/bYSGhhqPPvqo8dlnnxl5eXnGkiVLjIMHD1bpvFC7bub9VZX3FnArBgwYYPTs2dN48803jbCwMKO4uNgICwsz0tLSjAMHDhh/+9vfjIcffth46KGHDMP4/p+HuDUEIh9R8QPbMAyja9euxsiRIw3DqByInn76aePhhx/2eO2kSZOM2NjY64790ksvGf7+/sbXX39tblu/fr3h5+dnnDhxwjx+s2bNjCtXrpg1jz/+uPHkk08ahmEYf//73w1Jxvbt2839//rXv4ygoCDj/fffNwzj34FIknH48GGzZsGCBUZERIS53rJlSyMjI8OjvxkzZhgul+uavefk5BiSjKNHj15z//eNVxFs5s6d61GzevVqo0GDBsb58+cNwzCMkpISo169esb69es9XlcRiNLS0owWLVqY4e+H9gHvupn3V1XeW8CtKCoqMho3bmz4+fkZq1evNmbMmGEkJCR41Bw7dsyQZOTl5X3vz0PcGm6Z+aBXX31Vy5Yt04EDByrtO3DggLp16+axrVu3bjp06JDKysquO2bTpk119913m+sul0vl5eXKy8szt7Vt21b+/v7memRkpIqLi83j1qlTR126dDH3N2rUSK1bt/boMzg4WC1btrzmGOfPn9eRI0c0atQoNWjQwFxefvllj1tP39W+fXv17t1bcXFxevzxx/Xee++Zc0B+yHj333+/x3q/fv1Ut25d/fnPf5b071sldrtd8fHx1+wjNzdX3bt3V926dSvtq8p5wXuu9/6q6nsLqKrw8HD97Gc/U0xMjAYOHKgvvvhCn3zyicfPkTZt2kiSjhw5csOfh7h1zBj1QT169FBiYqLS0tI0YsSIWjvu1b/sbTabysvLb3kM4//+Osy5c+ckSe+9955HsJLkEcSu3p6ZmakdO3Zo06ZNmj9/vn71q19p165dCg4Ovunx6tev77EeEBCgxx57TBkZGRo8eLAyMjL05JNPXncSdVBQ0HXPuSrnBe/x1vsLuJY6deqYP3fOnTunRx55RK+++mqlusjIyBv+PGzRokVtt37HIRD5qFdeeUUdOnRQ69atPbbHxMRo+/btHtu2b9+ue++994a/fAsKCnT8+HFFRUVJknbu3Ck/P79K419PTEyMrly5ol27duk//uM/JEnffPON8vLyFBsbe1NjREREKCoqSv/4xz80ZMiQm3qN9O9Q1a1bN3Xr1k1Tp05Vs2bNtHr1aqWkpFRpvApDhgzRww8/rP3792vLli16+eWXr1vbrl07LVu2TJcvX64U+qp6XvCea72/qvre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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = 'MultipleLines', data = df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* If a customer does not have a phone service, he/she cannot have multiple lines. \n",
    "* MultipleLines column includes more specific data compared to PhoneService column. \n",
    "* So I will not include PhoneService column as I can understand the number of people who have phone service from MultipleLines column. \n",
    "* MultipleLines column takes the PhoneService column one step further."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MultipleLines</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>No</th>\n",
       "      <td>0.250442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No phone service</th>\n",
       "      <td>0.249267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Yes</th>\n",
       "      <td>0.286099</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Churn\n",
       "MultipleLines             \n",
       "No                0.250442\n",
       "No phone service  0.249267\n",
       "Yes               0.286099"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['MultipleLines','Churn']].groupby('MultipleLines').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Contract, Payment Method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>customerID</th>\n",
       "      <th>gender</th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>PhoneService</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>...</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>Contract</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>TotalCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7590-VHVEG</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>29.85</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5575-GNVDE</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>1889.5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3668-QPYBK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>108.15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7795-CFOCW</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>1840.75</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9237-HQITU</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>151.65</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \\\n",
       "0  7590-VHVEG  Female              0     Yes         No       1           No   \n",
       "1  5575-GNVDE    Male              0      No         No      34          Yes   \n",
       "2  3668-QPYBK    Male              0      No         No       2          Yes   \n",
       "3  7795-CFOCW    Male              0      No         No      45           No   \n",
       "4  9237-HQITU  Female              0      No         No       2          Yes   \n",
       "\n",
       "      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \\\n",
       "0  No phone service             DSL             No  ...               No   \n",
       "1                No             DSL            Yes  ...              Yes   \n",
       "2                No             DSL            Yes  ...               No   \n",
       "3  No phone service             DSL            Yes  ...              Yes   \n",
       "4                No     Fiber optic             No  ...               No   \n",
       "\n",
       "  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \\\n",
       "0          No          No              No  Month-to-month              Yes   \n",
       "1          No          No              No        One year               No   \n",
       "2          No          No              No  Month-to-month              Yes   \n",
       "3         Yes          No              No        One year               No   \n",
       "4          No          No              No  Month-to-month              Yes   \n",
       "\n",
       "               PaymentMethod MonthlyCharges  TotalCharges Churn  \n",
       "0           Electronic check          29.85         29.85     0  \n",
       "1               Mailed check          56.95        1889.5     0  \n",
       "2               Mailed check          53.85        108.15     1  \n",
       "3  Bank transfer (automatic)          42.30       1840.75     0  \n",
       "4           Electronic check          70.70        151.65     1  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Contract', ylabel='count'>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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2GMbN26sZ555RiNGjFDt2rXVsGFDvfbaa177O3z4sO6//36FhoYqLCxM/fr106FDhy54bMuyFBsbq+eee85reXZ2thwOh/bv3y9JKigo0KhRo1S/fn05nU51795d27Zts8cfOHBA/fr1U0REhIKDg9WxY0etWbPGa5+NGzfWU089paFDh8rpdGrMmDE/52UDriuRkZH2IyQkRA6HQ5GRkQoMDNQNN9ygPXv2SJLKysoUFhamzp0729v+7W9/U0xMjP18x44d6t69uwIDA1W3bl2NGTNGRUVFFzwu7wG4IhZwlQ0bNszq16+f9cILL1g9evSwl/fo0cOaPXu21a9fP2vYsGGWZVnWI488YkVHR1sff/yxtXPnTmvYsGFWnTp1rG+//dayLMv65JNPLElWp06drIyMDGvnzp3WHXfcYd16662WZVnWyZMnrUcffdRq2bKllZuba+Xm5lonT560LMuyJFkNGjSwFi1aZO3bt8965JFHrODgYHvfP1ZcXGzNmTPHcjqd9r6OHz9uWZZl/fKXv7Ti4uKsdevWWdnZ2VZCQoIVGxtrlZSUXPR1ODf3zp07W+vXr7e++OILKzY21rrzzjutXr16WV988YW1bt06q27dutbMmTPt7SrSawJURN9++63lcDisZ5555oLrR48ebdWpU8cqKyuzLMuyGjVqZIWFhVnz5s2z9u3bZ82YMcOqVq2atWfPHsuyLKukpMSKi4uzRowYYW3fvt3atWuXNWjQIKt58+ZWcXHxBY/x9NNPW/Hx8V7LHnnkEatLly728549e1p9+/a1tmzZYn311VfWo48+atWtW9f+9y07O9tasGCBtWPHDuurr76ypkyZYtWsWdP697//be+jUaNGltPptJ577jlr//791v79+8v/wgHXsTfffNMKCQmxn7dr18569tlnLcv6/t+1sLAwy9/f3/7v+qhRo6zBgwdblmVZRUVFVlRUlHXfffdZO3bssNLT060mTZrY/69yIbwH4HIRW7jqzsVWfn6+FRAQYB06dMg6dOiQVbNmTevo0aN2bBUVFVk1atSwFi5caG9bUlJiRUdHW7NmzbIs6/+HxZo1a+wxy5cvtyRZp06dsizLsp544gmrTZs2581DkjVlyhT7eVFRkSXJWrFixUXn/uM3a8uyrK+++sqSZG3YsMFe9s0331iBgYHWkiVLLrqvC819xowZliTrwIED9rL/+Z//sRISEuw5VrTXBKhoNm7caEmy3n///Quuf+GFFyxJVl5enmVZ3//PyoMPPmivLysrs8LDw6358+dblmVZf/3rX63mzZvbcWZZ3//lS2BgoLVq1aoLHuO///2v5efnZ23atMmyrO//Pa1Xr56VmppqWZZlffrpp5bT6bROnz7ttd2NN95ovfrqqxc9t5YtW1ovvfSS/bxRo0bWvffee9HxAL734/9+JycnW4mJiZZlWdacOXOsBx54wGrTpo3937vY2FjrtddesyzLsl577TWrTp06VlFRkb398uXLrWrVqllut/uCx+M9AJeLjxHCmPr16ysxMVGpqal68803lZiYqHr16tnrDxw4oDNnzui2226zl9WoUUO/+MUvtHv3bq99tW7d2v5zVFSUJCk/P/8n5/DD7WrVqiWn02lv17JlSwUHBys4OFh9+vS56D52796t6tWrq1OnTvayunXrqnnz5vY8+/TpY++rZcuWF51DRESEgoKC1LRpU69l5+bk69cEqEwsy7rssT/85/7cR43O/XO/bds27d+/X7Vr17b/PQ4LC9Pp06d14MCBC+4vOjpaiYmJeuONNyRJH330kYqLi/XrX//a3mdRUZHq1q1r7zM4OFgHDx6091lUVKTHHntMcXFxCg0NVXBwsHbv3q2cnByvY3Xo0OHyXxQAkqQ777xT69evV2lpqTIzM9W1a1d17dpVGRkZOnLkiPbv36+uXbtK+v6/823atFGtWrXs7W+77TaVlZXZH9H/Md4DcLmq+3oCuL6NGDFC48ePlyTNmzev3PupUaOG/WeHwyHp+89gX8l257Y9t93HH3+sM2fOSJICAwPLPTdJev311+2bgfz4mD+e+6XmdCVMvCZAZRAbGyuHw6Hdu3frV7/61Xnrd+/erTp16qh+/fr2skv9c19UVKT27dtr4cKF5+3rh/v4sVGjRmnIkCGaPXu23nzzTT3wwAMKCgqy9xkVFaWMjIzztjt3x9PHHntMaWlpeu655xQbG6vAwEANGDDgvC/A//B/AAFcni5duuj48eP64osvtG7dOj3zzDOKjIzUzJkz1aZNG0VHR6tZs2Y/6xi8B+ByEFswqnfv3iopKZHD4VBCQoLXuhtvvFH+/v7asGGDGjVqJEk6c+aMtmzZogkTJlz2Mfz9/VVaWnrFczt3zJ/aV1xc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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,6))\n",
    "sns.countplot(x = \"Contract\", data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Contract</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Month-to-month</th>\n",
       "      <td>0.427097</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>One year</th>\n",
       "      <td>0.112695</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Two year</th>\n",
       "      <td>0.028319</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Churn\n",
       "Contract                \n",
       "Month-to-month  0.427097\n",
       "One year        0.112695\n",
       "Two year        0.028319"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Contract','Churn']].groupby('Contract').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* It seems like, as expected, **customers with short-term contract are more likely to churn.** \n",
    "* This clearly explains the motivation for companies to have long-term relationship with their customers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='PaymentMethod', ylabel='count'>"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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oKS6d8q9xV0CSogNAzcUdwlAicSRLQAAAACwgLAFAAAAABYQtgAAAADAgmINW9988406d+6s0NBQORwOLVq0yGV8z5495XA4XB7t27d3aXPy5El1795dfn5+CggIUO/evZWamurSZvv27br//vvl5eWlsLAwjRs3zvaiAQAAACjhivUGGefOnVODBg3Uq1cvde3aNdc27du316xZs5zPPT09XcZ3795dSUlJWrFihdLT0/Xss8/queee09y5cyVJKSkpateundq2bavp06drx44d6tWrlwICAvTcc8/ZWzgAAADka13zFsVdAkqIFt+su+7zLNaw1aFDB3Xo0CHPNp6engoODs513O7du7Vs2TJt2bJFTZo0kSRNmTJFDzzwgP75z38qNDRUc+bM0aVLlzRz5kx5eHioTp06iouL04QJEwhbAAAAAKy54a/ZWrt2rQIDAxUZGam+ffvqxIkTznGxsbEKCAhwBi1Jatu2rdzc3LRp0yZnm+bNm8vDw8PZJjo6WvHx8Tp16tT1WxAAAAAAJcoN/X+22rdvr65du6patWpKSEjQ3/72N3Xo0EGxsbFyd3dXcnKyAgMDXV5TqlQplS9fXsnJyZKk5ORkVatWzaVNUFCQc1y5cuVyzDctLU1paWnO5ykpKUW9aAAAAABucTd02HryySedf9erV0/169fX7bffrrVr16pNmzbW5jt27FiNGTPG2vQBAAAA3Ppu+NMIL1e9enVVrFhR+/btkyQFBwfr6NGjLm0yMjJ08uRJ53VewcHBOnLkiEub7OdXuxZs+PDhOnPmjPNx6NChol4UAAAAALe4myps/frrrzpx4oRCQkIkSVFRUTp9+rS2bt3qbLN69WplZWWpadOmzjbffPON0tPTnW1WrFihyMjIXE8hlH6/KYefn5/LAwAAAACuRbGGrdTUVMXFxSkuLk6SlJiYqLi4OB08eFCpqakaMmSINm7cqP3792vVqlXq0qWLIiIiFB0dLUmqVauW2rdvrz59+mjz5s3asGGDBgwYoCeffFKhoaGSpG7dusnDw0O9e/fWzp07NX/+fL333nsaPHhwcS02AAAAgBKgWMPW999/r0aNGqlRo0aSpMGDB6tRo0YaOXKk3N3dtX37dj344IOqUaOGevfurcaNG2v9+vUu/2trzpw5qlmzptq0aaMHHnhA9913n2bMmOEc7+/v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zpypJk2a5NjxupaaL7d//3516tRJffv21Ztvvqny5cvr22+/Ve/evXXp0iXnDlBuNV6+gzhr1iy9+OKLWrZsmebPn68RI0ZoxYoVatasWa515NfHXq3e3FSsWNF5LU22d999V++9954mTZrkvEZk0KBBLtuwm5tbjp3coroxg5T7d0xe3zuvvvqqVqxYoX/+85+KiIiQt7e3Hn300Wu+kUlB+oqTJ0/Kx8fH6o8ld9xxhxwOh/bs2VOg9ld+1gVdR2zI730p6HdEYfeRLlexYkWdP39ely5dkoeHhyTp3Llzio6OVnR0tObMmaNKlSrp4MGDio6OdtZwPddv6ff1Oa++rKB9TX4Kus6ePHnS5WCAbdwgo4S78847tXfvXgUGBioiIsLl4e/vr7Jly6pq1ap53oa1dOnSyszMzHdetWrV0qFDh3To0CHnsF27dun06dOF/qWlXr16ysrK0rp166y0z82dd96p+Pj4HO9XRESE3NzcVL9+ff3666957jS7ublp9uzZaty4sVq1aqXDhw8Xuh4Unfbt2+vSpUtKT0933iDgcidOnFB8fLxGjBihNm3aqFatWjl25PJz5513ateuXbmuPx4eHqpZs6YyMjK0detW52vi4+PzvB1xQbbTgtS1c+dOVa1aNUddPj4+uuOOO+Tt7Z3vPPr27au3335bDz744B/azm41DodDbm5uzjtefvfddwoPD9ff//53NWnSRHfccYcOHDhQqGnffvvtWrNmjb744gsNHDgwz7bZO2QF6bO3bt2qrKwsjR8/Xs2aNVONGjUK3Vc1atRIw4cP13fffae6detq7ty5V22bXx97rfPdtWuXy7ANGzaoS5cu+vOf/6wGDRqoevXqOfrrSpUqudzkIyUlRYmJiS5tcvvuq1WrljZs2JBjfn/kaEL2NHr27KmHH35Y9erVU3BwcI4bWXl4eOT7udavXz/fbfinn37KcWOgola+fHlFR0dr6tSpLjcHypbf7dfzW0ey9zcu/wwvv+lGbgq6beT1Hhb2O8Lf318hISHatGmTc9iV3wO5yb41++Xr+J49e3TixAm9/fbbuv/++1WzZs0cR8MrVaqks2fPurz3V97UKbf1ycZ+nFSwvqag63d++1/ZZ27ZXscvR9i6BaWlpSk5OdnlcfkdxS7XvXt3VaxYUV26dNH69euVmJiotWvX6sUXX9Sv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Z+rbb7/VyZMn9fnnn1/TZwkAKCADAEAR69Gjh+nSpYsxxpisrCyzYsUK4+npaV599dUcbT/99FNToUIF5/NNmzYZd3d3c/jwYWOMMUeOHDGlSpUya9euNcYYk5iYaCSZDz/80Pma//73v0aSWbVqlXPY2LFjTWRkpPP57bffbubOnesy79dff91ERUVddbo7d+40kszu3buNMcbMmjXL+Pv751iGFi1amJdeesnExcWZsmXLmtTUVLNu3ToTGBho0tPTTYMGDcyoUaOMMcYcOHDAuLu7m99++81lGm3atDHDhw/Pcz7h4eHmz3/+s/N5VlaWCQwMNNOmTTPGGDNjxgxTrlw5k5qa6myzdOlS4+bmZpKTk40xxoSEhJhx48Y5x6enp5vKlSs7Py8AQNEpVbxRDwBwq1qyZIl8fX2Vnp6urKwsdevWTaNHj9bKlSs1duxY7dmzRykpKcrIyNDFixd1/vx5lSlTRnfffbfq1Kmjjz76SH/961/1f//3fwoPD1fz5s1dpl+/fn3n30FBQZKkevXquQw7evSopN+P+CQkJKh3797q06ePs01GRob8/f2vOt3sa62OHj2qmjVr5rvMDRo00B133KHPPvtMa9as0dNPP61SpVy/anfs2KHMzEzVqFHDZXhaWpoqVKiQ7zwur8/hcCg4ONi5nLt371aDBg3k4+PjbHPvvfcqKytL8fHx8vLyUlJSkpo2beocX6pUKTVp0oRTCQHAAsIWAMCKVq1aadq0afLw8FBoaKhKlSql/fv3q1OnTurbt6/efPNNlS9fXt9++6169+6tS5cuqUyZMpKkv/zlL5o6dar++te/atasWXr22WflcDhcpl+6dGnn39njrhyWlZUlSc5rlj744AOXoCFJ7u7u+U43ezoF0atXL02dOlW7du3S5s2bc4xPTU2Vu7u7tm7dmmPevr6++U7/8vqya7yW+gAA1w/XbAEArPDx8VFERISqVKniPLqzdetWZWVlafz48WrWrJlq1Kihw4cP53jtn//8Zx04cECTJ0/Wrl271KNHjz9US1BQkEJDQ/XLL78oIiLC5VGtWrUCT8fDw0OZmZl5tunWrZt27NihunXrqnbt2jnGN2rUSJmZmTp69GiOWoKDgws8n9zUqlVL27Zt07lz55zDNmzYIDc3N0VGRsrf318hISHatGmTc3xGRoa2bt16zfMCAOSPsAUAuG4iIiKUnp6uKVOm6JdfftHs2bM1ffr0HO3KlSunrl27asiQIWrXrp0qV678h+c9ZswYjR07VpMnT9bPP/+sHTt2aNasWZowYUKBp1G1alWlpqZq1apVOn78uM6fP59r7UlJSVq1alWu06hRo4a6d++uZ555RgsXLlRiYqI2b96ssWPHaunSpQWeT266d+8uLy8v9ejRQz/99JPWrFmjgQMH6umnn3aeavnSSy/p7bff1qJFi7Rnzx7169dPp0+fLvB7AAAoOMIWAOC6adCggSZMmKB33nlHdevW1Zw5czR27Nhc22afWpjb3fwK4y9/+Ys+/PBDzZo1S/Xq1VOLFi0UExNzTUe27rnnHr3wwgt64oknVKlSJY0bNy7XdgEBAS7XTV1p1qxZeuaZZ/TKK68oMjJSDz30kLZs2aIqVapc03yuVKZMGX399dc6efKk7rrrLj366KNq06aN/vWvfznbvPLKK3r66afVo0cPRUVFqWzZsnr44YcL/B4AAArOYbgiFgBwA5o9e7ZefvllHT58WB4eHsVdDgAA14wbZAAAbijnz59XUlKS3n77bT3//PMELQDATYvTCAEAN5Rx48apZs2aCg4O1vDhw4u7HAAACo3TCAEAAADAAo5sAQAAAIAFhC0AAAAAsICwBQAAAAAWELYAAAAAwALCFgAAAABYQNgCAAAAAAsIWwAAAABgAWELAAAAACwgbAEAAACABf8fyaoOi3lM19gAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,6))\n",
    "sns.countplot(x= \"PaymentMethod\", data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Bank transfer (automatic)</th>\n",
       "      <td>0.167098</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Credit card (automatic)</th>\n",
       "      <td>0.152431</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Electronic check</th>\n",
       "      <td>0.452854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mailed check</th>\n",
       "      <td>0.191067</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              Churn\n",
       "PaymentMethod                      \n",
       "Bank transfer (automatic)  0.167098\n",
       "Credit card (automatic)    0.152431\n",
       "Electronic check           0.452854\n",
       "Mailed check               0.191067"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Continuous Variables\n",
    "\n",
    "* The continuous features are **tenure**, **monthly charges** and **total charges.** \n",
    "* The amount in total charges columns is proportional to tenure (months) multiplied by monthly charges. \n",
    "* So it is unnecessary to include total charges in the model. Adding unnecassary features will increase the model complexity. \n",
    "* It is better to have a simpler model when possible. Complex models tend to overfit and not generalize well to new, previously unseen observations. \n",
    "* Since the goal of a machine learning model is to predict or explain new observations, overfitting is a crucial issue.\n",
    "\n",
    "Let's also have a look at the distribution of continuous features.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>customerID</th>\n",
       "      <th>gender</th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>PhoneService</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>...</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>Contract</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>TotalCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7590-VHVEG</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>29.85</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5575-GNVDE</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>1889.5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3668-QPYBK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>108.15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7795-CFOCW</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>1840.75</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9237-HQITU</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>151.65</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \\\n",
       "0  7590-VHVEG  Female              0     Yes         No       1           No   \n",
       "1  5575-GNVDE    Male              0      No         No      34          Yes   \n",
       "2  3668-QPYBK    Male              0      No         No       2          Yes   \n",
       "3  7795-CFOCW    Male              0      No         No      45           No   \n",
       "4  9237-HQITU  Female              0      No         No       2          Yes   \n",
       "\n",
       "      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \\\n",
       "0  No phone service             DSL             No  ...               No   \n",
       "1                No             DSL            Yes  ...              Yes   \n",
       "2                No             DSL            Yes  ...               No   \n",
       "3  No phone service             DSL            Yes  ...              Yes   \n",
       "4                No     Fiber optic             No  ...               No   \n",
       "\n",
       "  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \\\n",
       "0          No          No              No  Month-to-month              Yes   \n",
       "1          No          No              No        One year               No   \n",
       "2          No          No              No  Month-to-month              Yes   \n",
       "3         Yes          No              No        One year               No   \n",
       "4          No          No              No  Month-to-month              Yes   \n",
       "\n",
       "               PaymentMethod MonthlyCharges  TotalCharges Churn  \n",
       "0           Electronic check          29.85         29.85     0  \n",
       "1               Mailed check          56.95        1889.5     0  \n",
       "2               Mailed check          53.85        108.15     1  \n",
       "3  Bank transfer (automatic)          42.30       1840.75     0  \n",
       "4           Electronic check          70.70        151.65     1  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='MonthlyCharges', ylabel='Count'>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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mXTTTTfp3nvv1YgRI5SXl6c77rhDWVlZuvLKKyVJ+fn5uvjii3X99dfrqaeeksfj0ZIlS7RgwQJWxAcA9HsEewAAYKjHHntMd9xxh374wx+qurpaWVlZuvHGG7V06dJAm5/97GdqaWnRDTfcoMbGRs2cOVMrV65UZGRkoM3y5cu1ZMkSzZo1S2azWfPnz9eyZcuMeEsAAPQpgj0AADBUXFycHnnkET3yyCPHbWMymXT33Xfr7rvvPm6bpKQkrVixohcqBAAgtIX0HPvOzk7dcccdysvLU1RUlIYNG6Z77rlHfr8/0Mbv92vp0qXKzMxUVFSUZs+erX379vU4T319vRYuXKj4+HglJCRo0aJFcrlcff12AAAAAAAIupAO9vfff7+efPJJPf744yoqKtL999+vBx54QI899ligzQMPPKBly5bpqaeeUkFBgWJiYjR37ly1t7cH2ixcuFA7d+7UqlWr9Prrr2vt2rW64YYbjHhLAAAAAAAEVUgPxV+3bp2uuOIKXXrppZKkIUOG6C9/+Ys2bNggqau3/pFHHtHtt9+uK664QpL0/PPPKz09Xa+++qoWLFigoqIirVy5Uhs3btSUKVMkdc3lmzdvnh566CEW1AEAAAAAhLWQ7rE/++yztXr1au3du1eStHXrVn3wwQe65JJLJEnFxcWqrKzU7NmzA69xOByaPn261q9fL0lav369EhISAqFekmbPni2z2ayCgoJjXtftdsvpdPZ4AAAAAAAQikK6x/7nP/+5nE6nRo0aJYvFos7OTv3qV7/SwoULJUmVlZWSpPT09B6vS09PDxyrrKxUWlpaj+NWq1VJSUmBNp9233336a677gr22wEAAAAAIOhCusf+r3/9q5YvX64VK1Zo8+bNeu655/TQQw/pueee69Xr3nbbbWpqago8SktLe/V6AAAAAACcqpDusb/lllv085//XAsWLJAkjRs3TocPH9Z9992na6+9VhkZGZKkqqoqZWZmBl5XVVWliRMnSpIyMjJUXV3d47xer1f19fWB13+a3W6X3W7vhXcEAAAAAEBwhXSPfWtrq8zmniVaLBb5fD5JUl5enjIyMrR69erAcafTqYKCAs2YMUOSNGPGDDU2NqqwsDDQZs2aNfL5fJo+fXofvAsAAAAAAHpPSPfYX3755frVr36lnJwcjRkzRh9//LEefvhhffe735UkmUwm3XTTTbr33ns1YsQI5eXl6Y477lBWVpauvPJKSVJ+fr4uvvhiXX/99Xrqqafk8Xi0ZMkSLViwgBXxAQAAAABhL6SD/WOPPaY77rhDP/zhD1VdXa2srCzdeOONWrp0aaDNz372M7W0tOiGG25QY2OjZs6cqZUrVyoyMjLQZvny5VqyZIlmzZols9ms+fPna9myZUa8JQAAAAAAgiqkg31cXJweeeQRPfLII8dtYzKZdPfdd+vuu+8+bpukpCStWLGiFyoEAAAAAMBYIT3HHgAAAAAAfD6CPQAAAAAAYYxgDwAAAABAGCPYAwAAAAAQxgj2AAAAAACEMYI9AAAAAABhjGAPAAAAAEAYI9gDAAAAABDGCPYAAAAAAIQxgj0AAAAAAGGMYA8AAAAAQBgj2AMAAAAAEMYI9gAAAAAAhDGCPQAAAAAAYYxgDwAAAABAGCPYAwAAAAAQxgj2AAAAAACEMYI9AAAAAABhjGAPAAAAAEAYI9gDAAAAABDGCPYAAAAAAIQxgj0AAAAAAGGMYA8AAAAAQBgj2AMAAAAAEMYI9gAAAAAAhDGCPQAAAAAAYYxgDwAAAABAGCPYAwAAAAAQxgj2AAAAAACEMYI9AAAAAABhjGAPAAAAAEAYI9gDAAAAABDGrEYXgNPj9/vV3O5VfWuHYu1WJcfYjC4JAAAAANCHCPZh7Ehjm1btqlJTmyfwXFK0TUMjGYgBAAAAAAMFwT5MbTxUr/UH6uSXZDZJjqgIOdu6eu7rW61KnP19eX1+o8sEAAAAAPQygn0YKqpwat2BOknSqIw4nT8yVXarRW5vpz4uaVRBcb3iJ1+m365v0F/O9MtsNhlcMQAAAACgtzBmO8zUudxas7takjRtSJLmjsmQ3WqRJNmtFp01NFkzUjzyeztUcMStR1bvM7JcAAAAAEAvO6VgP3ToUNXV1X3m+cbGRg0dOvS0i8Kx+Xx+vbmjUl6fX9lJUZo+NOmY7bKi/apb+bgkadnqfVq5o7IvywQA9APc6wEACB+nFOwPHTqkzs7Ozzzvdrt15MiR0y4Kx1ZU6VRdS4ciI8y6eEyGzKbjD7Fv2blGl42IkST98h/b1dja0VdlAgD6Ae71AACEj5OaY//aa68F/vzWW2/J4XAEfu7s7NTq1as1ZMiQoBWH//D6fCoorpckTR2SpGjbF/+r+/b4OO1pMmlftUu/eXO3fjN/fG+XCQAIc9zrAQAIPycV7K+88kpJkslk0rXXXtvjWEREhIYMGaLf/va3QSsO/7HjiFPN7V7F2C0aP8jxxS+QFGEx6ddfHaevP7VeL24s1fzJgzV1yLGH7wMAIHGvBwAgHJ1UsPf5fJKkvLw8bdy4USkpKb1SFHry+fzadLirt37akCRZLSc+g2LqkCQtmJqtFzeW6t7Xd+nVxefI9DlD+AEAAxv3egAAws8pzbEvLi7mRt+Hiuta1OLuVFSERWOyTqy3/pP+e+5IRdss2lrWpLd3VfVChQCA/oZ7PQAA4eOU97FfvXq1Vq9ererq6sC3+92eeeaZ0y4M/7HjSJMkaXRmvCynsCd9Sqxdi2bm6bE1+/Xbt/dodn76KZ0HADCwcK8HACA8nFKP/V133aU5c+Zo9erVqq2tVUNDQ48Hgqe53aPDda2SpDFZ8ad8nu+dO1SOqAjtrXLpta2sZgwA+Hzc6wEACB+n1GP/1FNP6dlnn9W3v/3tYNeDT9lV4ZRf0qCEKCXG2E75PI6oCN1w3lA9+NYePfnvA7piwiCZ6bUHABwH93oAAMLHKfXYd3R06Oyzzw52LfgUv9+voopmSdLY0+it7/btGbmKs1u1t8qlf++tPu3zAQD6L+71fafO5daO8ia1ezqNLgUAEKZOKdh/73vf04oVK4JdCz6l1tWhpjaPLGaThqXFnvb54iMjdNVZOZKkp947eNrnAwD0X9zre1+ty60VBSV6oaBEq4uqtWJDicob24wuCwAQhk5pKH57e7uefvppvfPOOxo/frwiIiJ6HH/44YeDUtxAd6DGJUnKTYpWxElscfd5vntOnp75oFgbiuu1uaRBZ+YkBuW8AID+hXt97/J0+vSv7RVqaPXIbJIiIyxqbvfqb5vL9OXxWRqSEmN0iQCAMHJKwX7btm2aOHGiJGnHjh09jrFHevB0B/thqaffW98tPT5SX5k0SH/dVKZnPijWmVcR7AEAn8W9vnd9uL9WDa0exdgt+tbUHFktJq3aVaUDNS36994afTspmh1sAAAn7JSC/bvvvhvsOvApTW0e1bo6ZDJJeanB/db+O2fn6a+byrRyR6WqnO1Kj48M6vkBAOGPe33vOdLQpq1lXVvZXpSfrhh718exOaMz9Nz6Q2pq82hbWaMmMaoOAHCCgjO+G0F3oLqrt35QQpSiIixBPfforHhNG5Ikr8+vFQUlQT03AAD4fB+Xdm0XODozXrnJ//ny3mY1a8bQZElSQXG92lhMDwBwgk6px/6CCy743GF4a9asOeWC0OVgbYskaXgQh+F/0jVn52rDoXqt2FCixRcMl83KdzwAgP/gXt87WtxeFR+9x0/KSfjM8dFZ8dpa1qhaV4d2HGnS1CFJfVwhACAcnVKw755z183j8WjLli3asWOHrr322mDUNaB1eH2qaOpaFbe3Fs+ZOyZDaXF2VTe79eaOCl0xcVCvXAcAEJ641/eOogqnfH4pIz5SKbH2zxw3m0yakJ2g1UXV2l3RrCm5iaxpAAD4QqcU7H/3u98d8/k777xTLpfrtAqCdKSxTT6/5IiKkCMq4otfcAoiLGZ9a1qOHl29Ty9tLCXYAwB64F4ffH6/XzvKnZKksYPij9tuRFqs/r2nRvWtHapudrMWDgDgCwV1/PXVV1+tZ555JpinHJBK6lslSdmJUb16na9PGSyTSVp3oE4lda29ei0AQP/Avf7UHWlsU1ObRzaLWWekxx23nd1q0dCjI/Z2Vzb3VXkAgDAW1GC/fv16RUbyrfLpKj0a7HOSonv1OoMTo3XuiFRJ0l83lfbqtQAA/QP3+lPXvX7OsLQYRVg+/yPYqMyu4L+nslk+n7/XawMAhLdTGor/1a9+tcfPfr9fFRUV2rRpk+64446gFDZQtbi9qmvpkCQN7uVgL0kLpmZr7d4avVxYqptmj5D1Cz5oAAAGBu71wdc9Om5I8hevn5ObFKOoCIvaPJ0qbWjtsXo+AACfdkrB3uFw9PjZbDZr5MiRuvvuuzVnzpygFDZQlTZ03fTT4uxB3+buWGbnpyspxqYqp1vv7a3RrPz0Xr8mACD0ca8PruZ2j+paOmTSiY3Is5hNGpoao53lTh2qI9gDAD7fKQX7P//5z8GuA0cF5tf3QW+91LVn7lcnDdL/flCsFzeWEuwBAJK41wfb4aP39/T4SEWe4Bf3uUnR2lnuDHw2AADgeE4p2HcrLCxUUVGRJGnMmDGaNGlSUIoayMob2yVJg3t54bxP+ubUbP3vB8Vas7ta1c3tSotj7iQAoAv3+uDoHoafm3ziX9xnJ0XLJKm+pUPN7Z5eqgwA0B+cUrCvrq7WggUL9O9//1sJCQmSpMbGRl1wwQV68cUXlZqaGswaB4wWt1dNbV037kxH34XrEelxmpybqMLDDfp74RH94PxhfXZtAEBo4l4fPD6fP9DrfjLBPjLCovT4SFU621VS3yrHF78EADBAndJKaT/60Y/U3NysnTt3qr6+XvX19dqxY4ecTqd+/OMfB7vGAaO8qU2SlBJrk93a+/PrP+mbU7MlSS9tLJHfz+q7ADDQca8Pnqrmdrm9PtmtZqWf5Ki4nKNfBLAtLQDg85xSsF+5cqV+//vfKz8/P/Dc6NGj9cQTT+jNN98MWnEDTcXRYfiZjr4bht/t0nGZirVbdaiuVQXF9X1+fQBAaOFeHzzd9/dBCVEym00n9drco2vulNS3iu/dAQDHc0rB3ufzKSIi4jPPR0REyOfznXZRA1V3j31WQt/PcY+xW3XZ+ExJ0t8Ky/r8+gCA0NLX9/ojR47o6quvVnJysqKiojRu3Dht2rQpcNzv92vp0qXKzMxUVFSUZs+erX379vU4R319vRYuXKj4+HglJCRo0aJFcrlcQa/1ZFU4u4J9xilMs0uPj5TNYla716dGz8l9KQAAGDhOKdhfeOGF+q//+i+Vl5cHnjty5Ih+8pOfaNasWUErbiDxdPpU0+yWJGUZ0GMvSV+bPFiS9K/tFWpxew2pAQAQGvryXt/Q0KBzzjlHERERevPNN7Vr1y799re/VWJiYqDNAw88oGXLlumpp55SQUGBYmJiNHfuXLW3twfaLFy4UDt37tSqVav0+uuva+3atbrhhhuCWuupqGzqHpF38sHeYjYFvvCvcxPsAQDHdkrB/vHHH5fT6dSQIUM0bNgwDRs2THl5eXI6nXrssceCXeOAUOVsl88vxdgtios8rc0KTtnk3ETlpcSotaNTb+6oNKQGAEBo6Mt7/f3336/s7Gz9+c9/1rRp05SXl6c5c+Zo2LCuxVz9fr8eeeQR3X777briiis0fvx4Pf/88yovL9err74qSSoqKtLKlSv1v//7v5o+fbpmzpypxx57TC+++GKPLyf6WqtXcrm9Mqmr9/1UZCZ0feFPsAcAHM8pJcjs7Gxt3rxZ77zzjnbv3i1Jys/P1+zZs4Na3EBSfvTb/CxHlEwmY27cJpNJX5s8WA++tUd/KywN9OADAAaevrzXv/baa5o7d66+/vWv67333tOgQYP0wx/+UNdff70kqbi4WJWVlT2u7XA4NH36dK1fv14LFizQ+vXrlZCQoClTpgTazJ49W2azWQUFBfrKV77ymeu63W653e7Az06nM+jvraGj656eEmtXhOWU+lOU5ejusT+11wMA+r+TukOsWbNGo0ePltPplMlk0kUXXaQf/ehH+tGPfqSpU6dqzJgxev/993ur1n7tdIbpBdNXJg2SySR9dLCeFXgBYAAy4l5/8OBBPfnkkxoxYoTeeust/eAHP9CPf/xjPffcc5KkysquUWTp6ek9Xpeenh44VllZqbS0tB7HrVarkpKSAm0+7b777pPD4Qg8srOzg/q+pP+E8VOZX98tPT5SJpPU1mmSJS4lWKUBAPqRkwr2jzzyiK6//nrFx8d/5pjD4dCNN96ohx9+OGjFDRR+v19VRxfWOdVhesGSlRClmcO7PjT8fTOL6AHAQGPEvd7n8+nMM8/Ur3/9a02aNEk33HCDrr/+ej311FNBvc6n3XbbbWpqago8SktLg36N+qM99qcT7CMsZqXG2iVJ9kGjglIXAKB/Oalgv3XrVl188cXHPT5nzhwVFhaedlEDjcvtVWtHp0wmKTXObnQ5gSH4f99cJp+PvXUAYCAx4l6fmZmp0aNH93guPz9fJSUlkqSMjAxJUlVVVY82VVVVgWMZGRmqrq7ucdzr9aq+vj7Q5tPsdrvi4+N7PILKbAkMxc88zS/uu0f02Qflf0FLAMBAdFLBvqqq6phb33SzWq2qqak57aIGmipn1/y+5BjbKc+/C6a5YzIUZ7eqrKGNPe0BYIAx4l5/zjnnaM+ePT2e27t3r3JzcyVJeXl5ysjI0OrVqwPHnU6nCgoKNGPGDEnSjBkz1NjY2ONLhzVr1sjn82n69OlBrfdE2VKHyOc3yW41KyH6+H+nJyLz6I459iyCPQDgs04qRQ4aNEg7duw47vFt27YpMzPztIv6pP68r2236ubQGIbfLTLCossmZEmSXi4M/rBEAEDoMuJe/5Of/EQfffSRfv3rX2v//v1asWKFnn76aS1evFhS1+KuN910k+6991699tpr2r59u6655hplZWXpyiuvlNTVw3/xxRfr+uuv14YNG/Thhx9qyZIlWrBggbKysoJa74mypQ+VJKXF2U97YdzuHntb+lC5vYymAwD0dFLBft68ebrjjjt67Bnbra2tTf/zP/+jyy67LGjF9fd9bbt199inx4VGsJf+Mxz/ze2VcrGnPQAMGH19r5ekqVOn6h//+If+8pe/aOzYsbrnnnv0yCOPaOHChYE2P/vZz/SjH/1IN9xwg6ZOnSqXy6WVK1cqMvI/987ly5dr1KhRmjVrlubNm6eZM2fq6aefDmqtJyMiNU9S14r4pysu0qpIi18mi1UHGzynfT4AQP9yUtvd3X777XrllVd0xhlnaMmSJRo5cqQkaffu3XriiSfU2dmpX/7yl0Er7pP72nbLy8sL/PnT+9pK0vPPP6/09HS9+uqrWrBgQWBf240bNwa2wHnsscc0b948PfTQQ4Z9i9/tkwvnpcUbP7++25k5CRqaEqODtS361/YKfWNK8FcKBgCEnr6+13e77LLLPvcLA5PJpLvvvlt33333cdskJSVpxYoVQa/tVNnSjgb7IKyfYzKZlGjzq6LNRLAHAHzGSfXYp6ena926dRo7dqxuu+02feUrX9FXvvIV/eIXv9DYsWP1wQcffGYrmtPx2muvacqUKfr617+utLQ0TZo0SX/84x8Dx79oX1tJX7iv7bG43W45nc4ej97S1OaR2+uTxWQKyjf6wWIymTT/aK/93wpZHR8ABoq+vtf3V36/XxFHg31qkO7vCRFdQ/APNhLsAQA9nVSPvSTl5ubqX//6lxoaGrR//375/X6NGDGix/D4YOne1/bmm2/WL37xC23cuFE//vGPZbPZdO211/bqvrZ33XVX0N/PsXQPw0+Js8liPr35d8E2/8zB+u3be7ShuF6H61qUmxxjdEkAgD7Ql/f6/qq21SdLZKxM8isx5vQWzuuWYPNJsugAPfYAgE856WDfLTExUVOnTg1mLZ/h8/k0ZcoU/frXv5YkTZo0STt27NBTTz2la6+9tteue9ttt+nmm28O/Ox0OpWd3TtD0bsXzksLofn13TIckZo5IlVr99bo74VlunnOSKNLAgD0ob641/dXh5q6wndchF9Wc3B2vEm0dfXYlzm9auvoVJTNEpTzAgDCn/F7q32Ofruv7SfUNHf12IfS/PpP+s+e9kfY0x4AgBN06Ohw+e7h88EQaZE6Wxrk80u7K3tvmiAAIPyEdLDvr/vadvP7pRpXV7AP1vy7YJszOl1xkVYdaWzTRwfrjC4HAICwcKixa0cZhy14wd5kkjqqDkiSdhxpCtp5AQDhL6SDfX/d17ZbW6fU7vHJZJKSY2yG1nI8kREWXX50T3sW0QMA4MR099g7gthjL0kdlfslSTuO0GMPAPiPkA72/XVf225Nnq7F8pKibbJaQvdfxdePDsf/144KNbezYA8AAJ+ntcOrSlenpOD22EuSu7vHvpweewDAf5zy4nl9pT/ua9utqaMr2Adjf9veNDE7QcNSY3SgpmtP+29OzTG6JAAAQtbuymb5JXld9Yq0xAb13B2VXcF+b1Wz3N5O2a0soAecrpKSEtXW1gb1nCkpKcrJ4TMz+k7IB/v+rLGjq5c+VOfXdzOZTPra5Gzdv3K3/lZYRrAHAOBz5GfE61cXJuv6xXdLo38e1HN3OqsVazPJ1eHXviqXxg5yBPX8wEBTUlKiUfn5amttDep5o6KjtbuoiHCPPkOwN1Dj0aH4qSHeYy9JX5k0SA++tVsbDzXoUG2LhqSwpz0AAMcSZbMoP8Wmtv0beuX8OY4I7arp0J7KZoI9cJpqa2vV1tqqhbc+qPScYUE5Z1XJAS2//xbV1tYS7NFnCPYGMdmi1OI9OhQ/NjQXzvukDEekzh2Rqvf21ujvm8v0U/a0BwDAELkOq3bVdLDlHQac3hgyX1RUJElKzxmmwSPGBPXcQF8i2BvEljpEkhRrtyraFh7/Gr4+ZXBXsC8s002zz5DFbDK6JAAABpxcR4Skrrn8wEDRW0Pmu7lcrl45L9BXwiNR9kMRaUMlhUdvfbfZ+emKj7SqvKld6w/UaeaIFKNLAgBgwMlN6Pr4RrDHQNIbQ+YlqWjDe3rzuUfV3t4etHMCRiDYG6S7xz4lxBfO+6TICIu+PDFLL3xUor8VlhLsAQAwQE5818e3mma36lxuJYfRZwngdAV7yHxVyYGgnQswUuhunt7PRaR2LaQRTsFekr42OVuS9OaOSjW1sqc9AAB9LSrCrJykaEnSHnrtAQAi2BvC7/fLlpIrSUoOo6H4kjRhsEOjMuLk9vr0981lRpcDAMCANDIjThLD8QEAXQj2Bqhr88kcGSuT/EqMDq9gbzKZtHB612iDFRtK5Pf7Da4IAICBJz8Q7FkZHwDAHHtDlDR1DWGPi/CH5cryV0wapF+9sUv7q11a/naBRqcG/8uJlJQU9v0EAOA4RmbES2IoPgCgC8HeACVNXklSfER49nY3VleoYes7ihozSz95/O+qff2hoF8jKjpau4uKCPcAABzDqMyuHvs9Vc3q9IVnRwEAIHgI9gY4HObBvra2Vo2b/qmoMbMUN+ZL+uacs2W3BO/8VSUHtPz+W1RbW0uwBwDgGIYkx8hmNavd49ORhjblJEcbXRKATykqKgrq+RjRis9DsDdA91B8R5gGe0nqqNyvhAifGj1mNUVl6czcRKNLAgBgwLCYTRqaEqPdlc3aV91MsAdCiLO+RpJ09dVXB/W8jGjF5yHY97FOn19lzvDuse82NM6nzfVmbT/SpEk5CTKZGAYIAEBfGZ4Wq92Vzdpf7dKs/HSjywFwVJura1HLS2/8pUaOnxyUczKiFV+EYN/HDtW1yOOTfB3tirGG96YE2dE+7Wgyq7HNo7KGNmUn0VsAAEBfGZ4WK0naV+0yuBIAx5KclavBI8YYXQYGiPBOlmFo79HVaz21JQr3Dm6r+T/76G4/0mRwNQAADCwj0rruwfsJ9gAw4BHs+9iequ5gf9jgSoJj3CCHJGl/jUvN7R6DqwEAYODo7rE/UO2S3x/e0/sAAKeHYN/HvnfuUN03K1nOja8aXUpQpMbZNTghSn6/tK2MXnsAAPrKkJRoWcwmNbu9qnK6jS4HAGAg5tj3sVi7VSOTbf2mx16SJuYkqKyxTduPNGlaXpIiLHxfBABAb7NbLcpNitbB2hbtr3YpwxFpdElA2PFJsiYNUkOnTXsqm+X1+eTzS1azSTarWfGREYqPtMoeEcS9nYFeQLDHactLiVF8pFXOdq92VzRr3GCH0SUBADAgDE+L1cHaFu2rbtbMESlGlwOEPG+nTyX1rSptaNORxjbVaqQGXf8HbXNL23ZWHvd1dqtZidE2ZTgileWIVFZClGLsRCmEDv5rxGkzm0yakJ2g9/fVaktpo8YOimfrOwAA+sDwtFi9vauKBfSAL1DpbNe2skYdqG5RR6fvE0fM8nW0K9ZuUZIjXlaLSWaTSV6fX25vp5xtXrV5OuX2+lTpbFels11bSrtemRAdoaEpMRqaGqtMR6TMfP6FgQj2CIoxWfH66GCd6ls7VFLfqtzkGKNLAgCg32PLO+Dzlda3av3BOlU0tQeei7VbNSQlWtmJ0arZ8YH+/ruf6Jq7/qCJk0cf8xyeTp+cbR7VuNyqaGpXRWO7al1uNbZ6tLmkUZtLGhUVYdGw1BjlZ8aLpSxhBII9gsJutWhMpkNbyhq1pbSRYA8AQB/o3vLuAMEe6KGhtUPv7anR4fpWSZLZJJ2RHqexWQ5lJUQGRpc27/BIXxDFIyxmJcfalRxr16iMeEmS29upkrpWHaht0aHaFrV5OrWj3Kkd5U5Fapgc53xL7T7m5aPvEOwRNBOyu4L9obpWNbR0KDHGZnRJAAD0a8PSur5Ir2vpUH1Lh5K492KA8/n82nS4QRsO1avT55fZ1LU985QhSYoN4px4u9WiEelxGpEep06fX0ca27S70qn91S61d9qUMHOhCtr9qt5arnGDHMpNjmaoPnoVwR5BkxBtU15KjIprW/RxaaMuHJVmdEkAAPRr0TarBiVE6Uhjm/ZXuzQtL8nokgDDNLd7tHJHpcqPDrvPTYrW+SNTlRDdu194Wcwm5SRFKycpWheM9Omdf6/V1kM1ihoyUcW1LSqubVF8pFXjBjk0Oite0TYiGIKPfckQVGfmJEiSdlU41eL2GlsMAAADQPc8exbQw0BW1tCqFQUlKm9ql81i1tzR6bpiYlavh/pPi7CYlSanql+6XVMjqzQpO0F2q1nOdq8+PFCnZz44pLd3Vaqm2d2ndaH/4+siBNWghChlxEeq0tmuj0sbNXM4W+8AANCbhqfF6r29NdpX3Wx0KYAhdpY3ac3uavn8UlqcXZeMzejzQH8s0eZOTTwjVTOGJWtvVbO2H2lSldOtoopmFVU0KzspSmfmJCo3KZodpXDaCPYIKpPJpKlDEvXPbRXaXtakKbmJioxg4RAAAHrLCHrsMYAVHm7QB/trJUlnpMXqotHpslpCa1ByhMWsMVkOjclyqKKpTR+XNGp/tUul9W0qrW9TUoxNk3ISNCo9LuRqR/gg2CPo8lJilBxrU52rQ9vKmpjvBwBAL+oeis/K+BhoPjpYp4LieknSlNxEnT0sOeR7vjMdUcocFyVnm0dbShu1s9yp+pYOrS6q1rr9dRo/2KHxgx3Mw8dJ4yshBJ3JZNLU3K4w/3FpgzydPoMrAgCg/+oO9uVN7XKxvg0GiE2H6wOh/uxhyTpneErIh/pPio+K0HlnpOq7M4fo3OEpirVb1ebpVEFxvZ758JBWF1WpsbXD6DIRRgj26BUj0mLliIpQu8enHUeajC4HAIB+KyHappRYuyR67TEwbD/SpA/310mSzhmWrKlDwnd0qN1q0Zm5ibru7CG6ZGyG0uPt6vT5taPcqefXH9abOypYaA8nhGCPXmE2mzQlN1GStLmkUV4fvfYAAPSW7nn2+wj26Ocq20x6d3e1pK7h91PCONR/ktls0hnpcfrmlGx97czBykuJkV/S3iqXVmwo0YfVVtkHjTK6TIQwJm+g14zKjNNHxXVyub3aXdGssYMcRpcEAEC/NDwtVusP1rGAHkJGSUmJamtrg3a+oqIiRaTkqqDWKr+k/Mw4nT0sOWjnDxUmk0mDEqM0KDFKNc1ubTpcr31VLlW2m5Vx9UO6/d063Rpbo/NGhNfUA/Q+gj16jdVs1uScRK3dV6uC4nqNymClTwAAesN/9rJnyzsYr6SkRKPy89XW2hq0c5rsMcq89hF5/SYNSojSrFHp/T7YpsbZdcnYTJ01tENrtx1UsdOnXTXStc9s0PjBDt180Rn60hmp/f7vASeGYI9eNW6QQ5tLGuVye7X9SJMm5SQaXRIAAP0OW94hlNTW1qqttVULb31Q6TnDTvt8fr/09r5GuSLTFGny6tLxmbKYB06YTYy2aXJypz548Hr94JG/6Z3idm0ra9J3/rxRU3IT9dM5IzWjH45ewMkh2PdjRUVFhp/XajFrel6SVu+u1sZDDRqT5ZDNSq89AADB1N1jX1LfKre3U3arxeCKACk9Z5gGjxhz2ufZdLherkib/F6PRsc2KipiYP733dlcp+smxuvOb8zQU+8d0PPrD2vT4QZ9648f6exhyfrpnDM0Obd/rDmAk0ew74ec9TWSpKuvvrpXr+NynVivQH5mvDYdblBTm0dbyho1rZ8scgIAQKhIjbMr1m6Vy+3V4bpWnZEeZ3RJQFBUOdu1/kDXCvj17/xBcV/9qsEVGS851q5fXjpa3zt3qH7/7n6t2FCidQfqtO7J9Zqdn65fXpqvvJQYo8tEHyPY90NtLqck6dIbf6mR4ycH/fxFG97Tm889qvb29hNqbzGbNGNoslburFTh4QaNH+RQ5AD9phUAgN5gMpk0LDVGW8uadLDGRbBHv+Dp9OmtnZXy+aVkOXV460qJYB+QHh+pu64Yq+vPG6rH1+zXy4VleqeoSu/trdY1M4boxxeOkCM6wugy0UcI9v1YclZuUIY/fVpVyYGTfs0Z6bHaeNimOleHCg836JzhKUGvCwCAgWxYaqy2ljXpQE2L0aUAQfHh/lo1tHoUY7douLtSm40uKEQNTozWb+aP1/fOzdOv3ijSu3tq9KcPivXK5jL95KIzdNW0HBawHgD4N4w+YTKZdPbQrkU9tpQ2qsXtNbgiAAD6l2FH59kfYAE99ANHGtq0taxJknRRfroi1GlwRaFveFqc/nzdND333WkakRarhlaPlv7fTl3y6PtadyB4Ww8iNBHs0WfyUmKUER8pr8+vjw7WGV0OAAD9ytCjc2oP1BDsEd68nT69U1QlSRqTFa/cZOaLn4wvnZGqN//rXN1z5VglRkdoX7VLV/2xQP/98lY1tHQYXR56CUPx0WdMJpPOHZGilwvLtKPcqXGDHUqLizS6LAAA+oXuHvuDNS3y+/3sbY2wVVBcr8Y2j2JsFp3L9M0eTmZ3qjE2admcJC3f3qy3DrTqb4VlentHub4zIU5fyo0K/I5ISUlRTk5Ob5WMPkKwR5/KSojSGemx2lvl0tq9tZp/5iA+eAAAEAS5ydEym6Rmt1c1zW6lxfPlOcJPfUuHNpc0SJLOH5kmOwsuSzr9Xa9sWaOUfPESOVOHaNmGJv1m+duqW/mYfK2NioqO1u6iIsJ9mCPYo8/NHJ6igzUtOtLYpv3VLo1g5V4AAE6b3WpRTlK0DtW1an+Ni2CPsOP3+/XvPdXy+aUhydEalsoQ/G7B2PXK55f2Or0qarIoesR0JYycpqG+I3rr/u+rtraWYB/mCPboc3GREZqcm6iC4nq9v79WeSkxrNQJAEAQDEuN1aG6Vh2oadHZwxjCjPCyr9ql0oY2WcwmnT8yjVGdx3C6u17lSJrocuutnZWqdXVotwYr+dKb1ebxBa9IGII0BUNMzk1UrN2q5navNpc0Gl0OAAD9wtCjPZwHWUAPYcbr8+nD/V0rt0/JTZQjiv3Xe0tKrF3fnJqtKbmJkvyKHXuhfvZOrfZWNRtdGk4DwR6GiLCYdc7wru3vNh6qV3O7x+CKAAAIf8NSj255x172CDPbSpvkbPcqxm7R5NxEo8vp96xms84ZnqIvpXnlba7VkeZOXfH4h3plc5nRpeEUEexhmJHpccp0dG1/997eGqPLAQAg7LGXPcJRu6dTGw7VS5JmDE1WBFM0+0xKpF8Vf/6xJqTb1Obp1M1/3apf/GO7OrwMzQ83/F8Dw5hMJl04Kk1mU1fPwn4+hAAAcFq6e+yPNLapraPT4GqAE7OhuF5ur0/JsTblZ8YbXc6A42tz6vZzk3TT7BEymaQVBSW69pkNamplRG04IdjDUCmx9sBwq3/vrZbby4cQAABOVVKMTQnRXXOTi2sZjo/Q19Tm0dayRknSucNTZGbBPENYzCbdNPsMPXPtVMXYLFp/sE5fefJDHa7j90i4INjDcNOGJMkRFaEWd6fWHagzuhwAAMLaf+bZMxIOoe/D/bXy+aWcpGjlJrO9ndEuGJWmv/3gbGU5InWwpkVXPvGhNh6dJoHQRrCH4awWs2aNSpMkbStrUp2bb2oBADhV3Xt/E+wR6iqb2rXv6FTMmcPZnjFU5GfG69XF52j8YIcaWj26+n8L9O891UaXhS9AsEdIyE6KVn5mnCRpc71FslgNrggA/sPv96vN0ym/3290KcAXYmV8hIuPDnaN1MzPjFNqnN3gavBJafGRevGGszRrVJrcXp+uf36T3t5ZaXRZ+BykJ4SMc0ek6lBtq5weKeGchUaXA2CAc7V7tbO8SXurXGpq86jT75fNYlZqnF3jBjl0RnqsTMwFRQgaejTYs5c9QllFU5sO17fKbJKm5yUbXQ6OIdpm1ZNXT9ZNL32sf22v1A+Xb9YjCybqsvFZRpeGYyDYI2RERVh04ag0vbG9QvHTv6qimg6daXRRAAYei1U7Gy3aW1os36c66Ds6fTrS2KYjjW3aeMimuWMy6GVCyOkein+wpkU+n19mM19AIfR8dLBr3nZ+ZrwcUREGV4PjsVnNWrZgkuzWbfrHx0f04798rA6vT189c7DRpeFTCPYIKcPTYpUb06nDLRYt29CoK873KtbOf6YA+kZ5s1eZ1z6i3U6LJCnTEanxgx3KckQp2mZRY5tHB2pc2lzSqLqWDr1cWKp5YzM1JIUFnxA6spOiFWExqc3TqQpnuwYlRBldEtBDeWObSo721k8dkmR0OfgCVotZv/36BNmtZr24sVT//fJWxUVG6KLR6UaXhk8gMSHkTEjs1IHyWlUpXff8c5fu/9p4o0sCMABsOlSv21bXypY6RHazX7NGZ2pEelyPNimxdqXE2jVhcILe2F6hsoY2vbatXJeNywwMfwaMFmExKzc5RvurXTpQ7SLYI+R8VNw9t57e+lBRVFT0hW2+NsSv6poorTnUpsXLN2npeckanWo7ZtuUlBTl5OQEu0x8DoI9Qk6EWap9/WFlLrxfL20q1ezR6XwjCKBXfbi/Vtc9u1EdXr/c5Xt16dQhGv6pUP9JkREWXTlxkFYVVWlPZbPe2lmlb061KSnm2B9wgL42LLUr2B+scem8M1KNLgcIONLYptL6NplNXVsew1jO+hpJ0tVXX31iLzCZlfqVX0ojpusXbx5W5Yqfy1Nz6DPNoqKjtbuoiHDfhwj2CEnusp26YmSMXt3Tolv/vk3jBp2rDEek0WUB6IcKD9fre89tUofXpymZdr3y8G2KPOsvX/g6i9mki/LT5Wr36khjm17bWq5vTcuW3Wrpg6qBz9c1gqSKlfERcgqOroQ/OjNe8fTWG67N5ZQkXXrjLzVy/OQTeo3XJ31Q41OdYjX0+sd0frpHMZ9IlVUlB7T8/ltUW1tLsO9DBHuErG+NjdNep0W7KpxasmKz/nLDWYqwsEMjgODZX92s7/x5o9o8nTrvjFQtHmfR3z3uE369xWzSvHEZenFjqZraPFq3v04XjErrxYqBE/OfLe9YGR+ho7yxTaUNXb31U/PorQ8lyVm5GjxizAm3zxjaqb8VlqmupUMbmmL1zSnZsln5nG4kgj1CVoTFpCevPlOXLftAmw436MG39ugX8/KNLgtAP9HQ0qFFz21Sc7tXU3IT9YerJ6tox9aTPk+0zaqL8tP1ysdHtO1Ik/Iz4xlhhF51InNhvfUdkqTd5Q3avHnzF7ZnPiz6wsZDXSvhj86MV3wkvfXhrHtK2osbS1Tf0qFVu6o0b1wG28AaiGCPkJabHKMHvz5e339hs55ee1BTchM1Z0yG0WUBCHPeTp9+sLxQh+talZ0UpaevmaIo26kPoc9OitaojDjtrmzW6t1V+tbUHLYYQ9CdzFxYkz1GOTe9pPo2n6bMmCl/R9vntmc+LHpbTbNbh+paZZI0OTfR6HIQBLGRVl06PlN/KyzT/hqXNh1uYJcDAxHsEfIuHpupRTPz9KcPivXTl7fqjYx45SRHG10WgDD2yDv79NHBesXarfrTtVODsujduSNSdKi2RbWuDhVVOjUmyxGESoH/ONm5sK+X+eX2mfSd36xQot1/3HbMh0Vf2HS4q7d+RFqsEqJZaLS/yHRE6fyRaVqzu1rrDtQpNc5OwDQIf+8ICz+/ZJQ+LmnQ5pJG3fhCof72/RmKYX97AKfgw/21euLf+yVJ9311nM74nNXvT0a0zaqpQ5L0/v5abSiu16iMeFnotUcvONG5sCnOMh1pbFNE8mANzozvg8qAY2tq82hfVdd6D1Po0e13xg1yqNrZrh3lTq3cUanz2YjDEKxwgLAQYTHr8avOVEqsTUUVTv3kpS3y+Y7f+wAAx1LT7NZNL22R3y99a1q2Lp+QFdTzjxvsULTNIme7V0UVzqCeGzhZiTFdc5jrWzsMrgQDXeHhBvkl5SZHKzXObnQ56AVfGpmqjPhIub0+baizSiZiZl/jbxxhIyshSn/49mTZrGa9vatKD7y1x+iSAIQRn8+vn768VTXNbp2RHqull5346r8nKsJiDswd3XCoXp18AQkDJR0d7tzQ6jG4EgxkLW6vdh39onNqLr31/ZXVbNa8cRmyW81q6DDLcfY3jS5pwCHYI6xMzk3SA/PHS5Keeu+AXt5UanBFAMLF0+8f1Nq9NYqM6BoBdDqL5X2e8YO6eu2b273aX81WYyfrN7/5jUwmk2666abAc+3t7Vq8eLGSk5MVGxur+fPnq6qqqsfrSkpKdOmllyo6OlppaWm65ZZb5PV6+7j60JLYHexb6LGHcT4ubVSnz69MR6SyEtgxpD+Li4zQBSO7tnx1nL1A++r43dOXCPYIO1dOGqQfXThckvSLf2xXwcE6gysCEOp2ljfpoaOjfP7n8jFBm1d/LFaLWeMGdS2ct7Wssdeu0x9t3LhRf/jDHzR+/Pgez//kJz/RP//5T7388st67733VF5erq9+9auB452dnbr00kvV0dGhdevW6bnnntOzzz6rpUuX9vVbCCmJRxeFbGz1yOdn9Aj6XodP2l7WJEmakpvIVmgDwMiMOGVHd8pktuiRgka1dgzsL1j7EsEeYekns8/QvHEZ8nT69b3nN2lXOXNZARyb29upn/51q7w+v+aOSdeCqdm9fs1xgxwym6SKpnZVO9t7/Xr9gcvl0sKFC/XHP/5RiYn/2QqrqalJf/rTn/Twww/rwgsv1OTJk/XnP/9Z69at00cffSRJevvtt7Vr1y698MILmjhxoi655BLdc889euKJJ9TRMXB7jOIirbKYTer0++VsYzg++t7BZrM6On1KjrEpLyXG6HLQRyYmdcrrrFGFq1O/eqPI6HIGDJYVR1gym0367dcnqqa5QBsPNeiaZwr08vfP5qYBDAAlJSWqra094fYvbHNqd2WL4u1mfXOYXx9//PFx2xYVBecDSIzdqhFpcdpT1awtZY2aMzojKOftzxYvXqxLL71Us2fP1r333ht4vrCwUB6PR7Nnzw48N2rUKOXk5Gj9+vU666yztH79eo0bN07p6emBNnPnztUPfvAD7dy5U5MmTerT9xIqzCaTEqMjVOvqUEOrhy3G0KdMVrv2N3dNeaK3fmCxmaW6fz2i9AW/0vKCEl00Ol3nHx2ij95DsEfYirJZ9L/XTtW3nv5Iuyqcuvp/C/S3H8xQpiPK6NIA9JKSkhKNys9XW2vrCbW3ZY1UxsIHZDJbtP/FezTr7vUn9DqX6/Tnxk/IdmhPVbP2Vrl03ohORUb0zpz+/uDFF1/U5s2btXHjxs8cq6yslM1mU0JCQo/n09PTVVlZGWjzyVDffbz72PG43W653e7Az05n/xv9lRhtOxrsO5QnvvxG34kZN1tun0nxkdZenf6E0NR+eKsuHRGtN/a1aun/7dTbP0nmPtjLCPYIa46oCD333Wn6xh/Wq7i2RVf/b4H+euMMJceylQrQH9XW1qqttVULb31Q6TnDPret1yetroyQy2tSTnSn5v/XLV94/qIN7+nN5x5Ve/vpD5/PiI9USmxXqNpT1awJgxNO+5z9UWlpqf7rv/5Lq1atUmRk3y6sdd999+muu+7q02v2NRbQgxG8Pr8c07vWwTgzN1FmM731A9FVY+NUWOVTSX2rfv/uft08Z6TRJfVrYTXHnpVycSypcXb9v0XTlOmI1IGaFn3rjx+pupk5rUB/lp4zTINHjPncx2FTmlxek2LsFl0yZcQXth88YoySMgYHrUaTyaT8zHhJ0u6K5qCdt78pLCxUdXW1zjzzTFmtVlmtVr333ntatmyZrFar0tPT1dHRocbGxh6vq6qqUkZG1xSHjIyMz9z7u3/ubnMst912m5qamgKP0tL+t9MKe9nDCB+UtMnqSJfd7NeYo78HMfBERZj1P5ePliQ99d5BHaxhp5jeFDbBnpVy8XkGJ0brhe9NV3q8XXurXPrmHz5SeWOb0WUBMEhpfau2lDZKkmbnpxs2/G9kepxMJqnS2U6P6XHMmjVL27dv15YtWwKPKVOmaOHChYE/R0REaPXq1YHX7NmzRyUlJZoxY4YkacaMGdq+fbuqq6sDbVatWqX4+HiNHj36uNe22+2Kj4/v8ehvAnvZt7B4HvqGz+fXP3a3SJKGx3XKagmbuIFecPHYDJ0/MlUdnT7d8X875GeHjl4TFv+nsVIuTsSw1Fj99cYZGpQQpeLaFn39qfUqqTuxebgA+g+3t1Orirp6a8dmxWtIsnHzimPs1sD1d1X0v/nbwRAXF6exY8f2eMTExCg5OVljx46Vw+HQokWLdPPNN+vdd99VYWGhrrvuOs2YMUNnnXWWJGnOnDkaPXq0vv3tb2vr1q166623dPvtt2vx4sWy2wf21KzuBfPaPJ1q93QaXA0GgtW7q1Xq9MrnbtGwOJ/R5cBgJpNJd395rOxWsz7cX6fXtpYbXVK/FRbB/pMr5X7SF62UK+m4K+U6nU7t3LnzmNdzu91yOp09HggPuckx+uv3Z2hIcrSONLbp639Ypz2VDIEFBpL399Wqud2r+Eirzh2RanQ5ys/oWjRqd2Wz6Kg4Nb/73e902WWXaf78+TrvvPOUkZGhV155JXDcYrHo9ddfl8Vi0YwZM3T11Vfrmmuu0d13321g1aHBZjUr1t61pFIDw/HRy/x+v37/7/2SpOaP/6WIsEga6G05ydFacsFwSdK9bxTJ2c4Iot4Q8ovnGbFS7kBYTKc/G5QQpb/eOEML/7dA+6pdmv/kOj1+1SS22QAGgOLaFu0s7/oy9qLR6bJZjf9UmZcSI5vFLJfbq7oOFpA6Ef/+9797/BwZGaknnnhCTzzxxHFfk5ubq3/961+9XFl4SoyJkMvtVX1LBzvHoFetP1inj0saFWGWnJv+T/rGFUaXBAN9cgvZafF+ZcVZVN7s1p0vfqirx5/81KeUlBTl5OQEs8R+JaSDvVEr5d522226+eabAz87nU5lZ2f32fVx+tLiI/XXG2foxhcKtaG4Xt99dqPu/PIYXTNjiKST3wf7ZPGLB+h7bZ5OvXN0CP6k7AQNTow2uKIuVotZQ1NjtLuyWUdajf+iAQNPUrRNpfVtamillwy964l3u3rrZ+VFa39Lo7HFwDDO+hpJ0tVXX93j+ajh05Q2f6n+tr1ey378NXU2153UeaOio7W7qIjP2McR0sH+kyvlduvs7NTatWv1+OOP66233gqslPvJXvtPr5S7YcOGHuf9opVy7Xb7gJ+T1x8kxtj0wqLp+sU/tutvhWVa+n87dbCmRdeMi9XYsaNPeB/sU8EvHqDv/XtPtVo7OpUYHaGzhyUbXU4Pw9NiPxHs6bVH32LLO/SFj0sa9OH+OlnNJl05KkZ/MLogGKbN1TVy7tIbf6mR4ycHnvf7pfeqfaqTXTNveUaTk0983Y+qkgNafv8tqq2t5fP1cYR0sO9eKfeTrrvuOo0aNUq33nqrsrOzAyvlzp8/X9KxV8r91a9+perqaqWldQ3FPpGVctE/2KxmPfi18RqaGqMHVu7Rs+sOaf2eCHVYY7Tw1ru+cB/sU8EvHqDv7atq1t4ql0wmac7ojJBbhTk3KVoRFpPaOiVb5gijy8EAkxhzNNgzxx69qLu3/spJg5QWw0KNkJKzcjV4xJgez81Kb9NfN5XpcItFM8fmKTmWztRgCelg371S7id9cqVcSYGVcpOSkhQfH68f/ehHx10p94EHHlBlZSUr5Q4wJpNJPzx/uIalxuq/X96qPXUeZV73mLwpdg0ewZc7QLhrcXu1Zk/XNmdTc5OU4ei7qVsnymoxKy8lRnurXIoeeY7R5WCASYzu2su+qc2jTp9fFjOjRhBcRRVOvVNULZNJ+sH5w9RUutfokhCiMh1RGpYaowM1LfrwQJ2+PCHL6JL6jdDq0jgFrJSLEzV3TIb+9eNzNSIpQpbIWH1UG6F/76mWp5OtWIBw5ff7taqoSu0en1Jj7ZqWl2R0Scc1Iq1rdfyYUTPZxxd9KtZuVYTFJJ9fcrYxzx7B191bP29cpoalxhpcDULdOcNTZDJ1LXh7pKHN6HL6jZDusT8WVsrF6chOita9FyTr4lt/L8e0r2prWZOKa1t04ag05Rq41zWAU/NxaaMO17XKYjZpzpj0kO6JzE2OlsXkV4e7VU1uvlBE3zGZTEqMtqm62a361o7A0HwgGA7WuPTG9gpJ0uLzhxtcDcJBYrRNY7Mc2n6kSe/vr9E3p2TLZArd+3e4CPsee+BkRVhManz3GZ2T6lGs3Spnu1evbinX27sq1eZhThgQLqqc7fpwf9fuFueNSFFKiM/Ti7CYdXGWRxV//pESIi1Gl4MBJjDPngX0EGRP/vuA/H5p1qg0jc46+S3MMDBNz0tShMWkKqdbxbUtRpfTLxDsMWBlRPn17bNyNWGwQ5JUVNGs59cd0pbSRnX6GCYLhDKPT3pzR6V8fml4aqzGDXIYXdIJIc/DKN3z7OtZQA9BVFzbolc+PiJJWnwhvfU4cTF2q8YPTpAkbThUzxS1ICDYY0CzWc06f2SavjFlsJJjbGr3+vTe3hq98NFh7a928UsGCFEf11vU1OZRXKRVs/LTGMIHfIGkoz32dS6CPYLnd6v2qtPn14Wj0nRmTqLR5SDMnJmTIKu5q9e+pL73tqEeKAj2gLpW6LxqWo4uHJWmaJtFjW0evbG9Qn/dVKbi2hYCPhBCYsfPUWmrRSaTdPGYDEVG0A0OfJGUmK6pKvUtHdzTEBS7K53657ZySdJP55xhcDUIR9E2q8YeHXG3oZhe+9NFsAeOMptNGjfIoWtnDNG0IUmymk2qdLbrta3l+suGUu2rbuYXDmCw3bUdSprzA0nSWUOTlZUQZXBFQHhwREXIYjLJ6/PL2e41uhz0A799e6/8funScZkakxUe06EQeibnJspiNqm8qV1HGlkh/3QQ7IFPsVnNmjEsWd85e4gm5yYqwmJSjcutf22v1HPrD2tzSYPaWWQP6HMVTW16YF2DTJYIDYryaWouwz6BE2U2m5QY0zXPvs7lNrgahLstpY1atatKZpP0k4vorcepi7VbNSaza9HFguJ6g6sJbwR74Dhi7FbNHJ6i687J07S8JNmtZjW1efT+vlr96YNirS6qUk0zH46AvtDu6dT3/1+hGtt96qgu1pRkL/PqgZOUfHQ4fh0r4+M0/fbtPZKkr0warOFp7FuP0zN5SKLMJqmsoU3l9NqfMoI98AWiIiyaMTRZi2bm6cJRaUqOtcnr82tHuVMrNpTo5cJS7a50ytvJvtRAb/D7/frFK9u1taxJsTaTal65V1buXsBJS4rtWkCvnmCP0/DRwTq9v69WVrNJ/zVrhNHloB+Ij4xQ/tFe+w2H6LU/VVajCwDCRYTFrHGDHBqbFa/yxnZtLWvU/hqXyhvbVd7YrvesNcrPjFcKUxeBoFq2er9e+fiILGaT/ntGoq67p8rokoCwlMzK+DhNfr9fv3lztyTpm1OzlZMcbXBF6C+mDknSrgqnDte1qqbZrdQ4u9ElhR2CPXCSTCaTBiVGaVBilFztXu0ob9LOcqdcbq8+Lm2UZFP6Vb/Re4dbNXpcJyt2A6fhuXWH9Lt39kqS7rx8tMbY+SYfOFXdwb6+tUM+v19mprPgJL22tVxbShsVbbPQW4+gckRFaHhqrPZVu7SltFEXjU43uqSww2BG4DTERlp11tBkXXfOEH15QpaGpsRI8isye6weLWjStF+9oztf26m9Vc1GlwqEnf/bckT/89pOSdJNs0fo2zOGGFsQEOYcURGymk3q9PnV1OYxuhyEmbaOzkBv/eILhistPtLgitDfnJnTtSjunspmtbgZAnuyCPZAEJhNJuWlxOjyCVmal+VR49r/p9Roi5ztXj277pDm/G6t5j+5Tn8rLFNbByvqA1/k3d3V+ulft0qSvnP2EHqGgCAwmUxKYjg+TtFT7x1QRVO7BiVEadHMPKPLQT+U4YhUpiNSnX6/tpU1GV1O2CHYA0EWZZWa1r+k389L1bPXTdXFYzJkMZtUeLhB//3yVk379Tta+n87VFThNLpUICSt3Vuj779QKK/Pr69MGqSll41mBXwgSALD8VlADyfhYI1LT/77gCTpF/PymWaIXjMpO0GStP1IEwtTnyTm2AO9xGI26fyRaTp/ZJqqne16ubBML24sUWl9m55ff1jPrz+sidkJWjg9R1+emCW7lZsksGpXlRYv36yOTp9m56fpga+Nl9lMqAeCpXtl/LoWtmvFifH7/br91R3q6PTpvDNSNW9chtEloR8blhqr+EirnO1e7a5s1thBDqNLChv02AN9IC0+UosvGK73/vsCvbBoui4dlymr2aQtpY265W/bdM5v3tVjq/fRg4IB7a8bS/X9FwrV0enTJWMz9PuFkxVh4TYFBFNgL3uG4uMEvbL5iNYdqJPdata9V4xlBBV6ldls0sSjvfYflzTK7/cbW1AYocce6ENms0kzR6Ro5ogU1TS79XJhqf7f+sOqaGrXb1ft1ePv7tf8yYP13XPyNDwt1uhy0U+VlJSotra2186fkpKinJycE27v8/n1yOp9WrZ6nyRp/pmDdf/8cbIS6oGg6x6K39DaoU6fXxZGxOBzVDS16c5/di1i+uNZI9jeDn1idFa8PjpYr/rWDh2ua9WQlBijSwoLBHuErKKiorA678lKjbPrh+cP1/XnDtW/tlfoj+8f1I4jTq0oKNGKghLNzk/TjV8apim5iXw7jqApKSnRqPx8tbW29to1oqKjtbuo6ITCvcvt1c0vbdHbu7r2pv/RhcN180Vn8N880EviIq2KsJjk6exaGb97MT3g03w+v255eZua272akJ2gG88banRJGCDsVovGDIrXxyWN2lzaQLA/QQR7hBxnfY0k6eqrr+7V67hcrl49/4mKsJh1xcRB+vKELG0ortcf3y/W6t1VeqeoWu8UVevMnATd+KVhuig/nbnGOG21tbVqa23VwlsfVHrOsKCfv6rkgJbff4tqa2u/MNjvKnfqR3/ZrAM1LbJZzLr3yrH6xtTsoNcE4D+6V8avcrpV53IT7HvRJ0dHub1+1bV1qqWjazGwSKtJydEWRUec3Mikkx0RdTqe+bBYH+yvVWSEWQ9/YwKjqNCnJg5O0JaSRpXWt6mBqaonhGCPkNPm6lot/tIbf6mR4ycH/fxFG97Tm889qvb29qCfu8d1TmFkQISkH44z64rcVL2216V3D7Vpc0mjbvx/hRoUZ9GXR8bq/Nwo+bwdstvtwS/6qL784ABjpOcM0+ARYwy5ts/n17PrDuk3b+5WR6dP6fF2PXX1ZE06un8tgN6VHGPvCvYtHWIjyd6x7+AhTbvyu7Jkj5d90BhFJGUds11nS4PcFfvkLtultgMb5ak9/LnnPZkRUadjQ3F9YM/6X87L17BUpgeib8VHRWhISoyKa1u07UiThtG39YUI9ghZyVm5vRI8qkoOBP2cnxTMEQeWmETFTb5ccZPm6Yhi9eSmJj3272I1b3pNzVvelN/dctrXOJa++uCAgWd/tUs///s2bTrcIElHV76fQK8h0IeSAyvj0wsWbNXN7frj2oNa/tEhOeb9tMcxi8kv+9FOb49P8vhNssQkKnr4NEUPn6bE87+jWKtfebGdyo3xyf6pzXJOZkTU6ahytmvxis3y+vz68oQsXX1Wbq9dC/g84wc7VFzboqIKp3LZjOELEeyBIOuNEQcen1Ts8mp/s0VtsUlKPP87Sv7S1RoWb9KIuE5FBfH/5L764BDOQm3xuXBQ53Lr0dX7tLygRJ0+v2JsFv18Xr6unp7DfHqgj3UvoFfnYsu7YGn3dOr37+7XH9YelNvbNdze66xW/qBkjR6arXRHpCKt5h6/7zq8PtW1uFXZ1K6S+laVNrTJ5ZW2N1q1s6lr268JgxOUlRDZZ78nne0eXfvMBtU0uzUyPU6/mT+O39EwTG5StBxREWpq86islakgX4RgD/SSYI84yJP0JZ9fb727VjtrPLKl5mpfs3TAZdHIjDhNzklUcmzvDc9Hl1BbfC7UtXV06s/rivX7dw/I5fZK6uqlv/PLYzQ4kdWVASN0b3nX2OaR1+czuJrQcDpf2O6p7dCyDY2qcHVKkkYkRWhGXIPu+eEiffOJv2vwcRb+slnNynREKdMRpUk5ierw+rS3qlnbjzSputmtfdUu7at2KSM+UmfmJiiyl3f9avd06vrnNml3ZbNS4+z64zVTFG0jKsA4JpNJ4wY59MH+Wh1oJth/Ef5vBcKIxWxSupr0zjM/06VLn1VD1GAdaWxTUUWziiqalZcSo8m5icpy9N23+wNNKC0+F8pqWzv1wMrdWrGhRI2tHknS2EHx+sW8fJ09LMXg6oCBLcZukc1qVofXF/j/cyA7nS9s46ZeqcQvfUcmi1Xe5jo1rH5ah/d8qHeOHj+ZhXptVrPGDnJo7CCHaprd2lbWqKLKZlU62/Wv7ZWKsUYoduIlcnuDn/Cd7R5979lN2nCoXnF2q569bipb2yEkjM6M1/qDdWr0mGXLPMPockIawR4IS34lW9yaNXmwKpraVHi4QQdqWlRc2/XIdERqcm6ihqbEEPB7iZGLz4WqTp9flW0mpXz5Z/r+G9Xy+aslSYMTo3TzRWfoyomD2NkBCAEmk0nJMTZVNLWrztWhgb6R1Kl8YevzS4V1FpW0dk2EHxzdqTMHxylizE8l/fS0F+pNjbNrVn66zhqarG1lTdpa1qgWr0/Jcxfrxjeq9b3mffr2WblKDML6JKX1rbrh/xWqqMKpOLtVz1w3VWOyHKd9XiAYomwWjUiL1e7KZsVNusTockIawR4Ic5mOKF02PkoNLR3aXNKgoopmVTS16/VtFUqMjtCZOYkalRknq5khTAg+T6dPRxratK/apQM1Lrm9EYrJP08+vzQ9L0nXnZOni0any0KgB0JKd7CvdbkHfLDvdqJf2HZ4fXpje4VKWltlMklfGpGq8YMdPb5ID9ZCvTF2q2YMS9bk3ESt27ZHhYfq5HSk6+FVe/Xkvw/om1OztWhmnrKTTr533e/36/VtFfrFK9vV7PYqJdau5787TaOz4oNSOxAs4wc7tLuyWdGjzlOzm+lDx0OwB/qJxBhb4Nv9LaWN2nakSQ2tHq3eXa31B+s0KTtB4wY5ZI+wfPHJgONo7fCqptmtI41tKmtoU5WzXb5PjAq1m/2q3fQvPX3rtZo/a7pxhQL4XClxXfPsa11u5ZLsT5in06fXtpbrSGObIiwmzRubqSHHmUMfTDarWcPjfHrtD9frkVfW6q3DPu2qcOrZdYf0/PpDmjcuUzecN1TjBjlOaKTeltJG3f/mbq0/WCdJmpybqEcXTGTtE4SkjPhIOSJ8apJdaw616kszjK4oNBHsgX4mxm7VOcNTNHVIknaUN+njkka53F59eKBOBcX1GpEeqzGZjj5dZRfhw9vpU0tHp1ztXrV0eOVydz3qWzpU0+xWa0fnZ14Ta7dqaEqMRqTHyl9TrN/d96TyfvM9A6oHcKJSY7uDfYfosj8x3k+EepvFrK9MGqQMR2TfFuH36dycKP34ikn6cH+d/rD2gN7fV6vXt1Xo9W0VykmK1gUjUzUpJ1FnpMcpJc4mu9WiFrdXxbUt2lLaqNe3VaioomsHH5vVrO9/aZh+fOFwWS2M7ENoMplMGhbn0+Z6s1YdbNWdfj+fYY+BYA/0UzarWWfmJGrC4ATtrWpW4eEG1bV0BBbaS4iO0JiseOVnxCvGzq+C/s7n86u2xa0DDR5FDZumg81mlR6ok8vtVYvbK1eHVy3tXrV7v3iIW0JUhDIdkRqUGKXBidGKj7QGbrBlvbcLIIAgSjka7F1ur9yf/b4On+L3+/XWziqVNXT11F85KavvQ/0nmEwmzRyRopkjUrSzvEl/XHtQ/9peqZL6Vj23/rCeW3/4c19vs5h12fhM3TznDHrpERYGR/u0qbJN5c1R2nioQdPykowuKeTwaR7o5yxmk/Iz4zUqI06VznbtOOLUvupmNbZ69OH+Oq0/UKfc5BidkR6roSmxRpeLU+D3+1Xr6tCRxjYdaWjTkcZWlTe2q7KpXZXOdlU721Xd7Jb36Jj5tK8t1ccNkhrqj3k+i9mkWLtVsXarYuwWxdqtckRFKDXOrpRYuyLo1QHCns1qDuwP3eSh5+uLvL+/VvtrXLKYTPryhCxlOqKMLilgTJZDjyyYpF99xav399Xqo4N12lbWqEN1rWpo7ZDfL0VYTMpwRGr8oASdNSxZl4/PVEL06S+8B/SVCLPUUrRWcRPm6sWNJQT7YyDYAwOEyWQK7Jf7pTNStbeqWTvLnap0tgdW07eYq5Vhtyp65Dm9sp0OTk9ja4cO1LToQI1LBwP/dKmsoU3uE+hpN5kkh92squI9GjJ0mFKSEnuE95ijYd5uNTPEDRgAUmPtamrzqLGD/98/z86j09okafbotJDt4Y6xW3Xx2AxdPDYj8Fynzy+vzye7lfV1EP5c295W3IS5+tf2Cv3P5WPkiIowuqSQQrAHBqBP7pVb39KhPVXN2lvV1Yt/pM2s1Ctv03WvVWnWvs2aPTpNF4xM45v9PmVSRbNXb2yr0M7yJu0sd2pXhVM1ze7jv8Ikpcd1DY8flBClrIQoZToilR4fqfR4uzIckUqNtWvb1i2aPHmernriFQ0ekdaH7wlAqEmJs2l/jdREsD+uKme73t1TI0k6a2iSRmWE14rxFrNJFjOhHv1DR/keZcdbVer06rWt5fr2WblGlxRSCPbAAJcUY9OMock6Ky9JNS63CosOaVdZndodaXpje4Xe2F4hi9mkKbmJumh0umbnp/fJCsADSbunUxVN7apoatOhKquyb3pJi9+skVTzmbaZjkgNS43V0NSYwD9zk2KU4YiUzcoQeQAnrnsBvUaG4h9Tm6dTb2yvUKfPr6EpMZo2hKG/gNFmD43Wn7c49dLGEoL9pxDsAUjqGqqfFhepcYmdWnX7d/Xi2+tV6kvUO0VV2l3ZrILiehUU1+veN4qUkxStmSNSdN6IFM0YlsJQqJPg9/vV0OpReVObKhq7wnxDq+cTLcwy26MVYZZGZzk0Oiu+65+Z8RqZEadYFjoEECTdW941e0yShd8tn+T3+7W6qErN7V45oiI0Z0w6U5SAEPCl3Cgt3+7SjiNO7TjSpLGDHEaXFDL4LQ7gmEYk2/TNM0fqv+eOVGl9q94pqtI7RVUqOFivkvpWrSgo0YqCEplN0oTsBJ07PEUzR6RqUk4Ci6t9QofXpypne6BHvqKp/Zjz4ROjI5TpiFKku16vP/hf+vDNv2vqlMkGVAxgoIg7uqaG2+tTRHKO0eWElO1HmnSgpkVmkzRvbAZz1IEQEW8366Ix6XpjW4Ve2lhKsP8Egj2AL5SdFK3rzsnTdefkyeX2quBgnd7fV6v399XoQE2LPi5p1McljVq2Zr+ibRZNyknQ1CFJmjYkSZNyEhVlGxgfiPx+v5raPKoMBPl21Ta79ellCK1mk9LjI5XpiFRmQqQy46MCf0dl+2rlqT0si5meIQC9y2QyKTXWrrLGNtnShxpdTshoaO3Q2n1de3eeMzxFafHGbWsH4LMWTM3WG9sq9OqWI/rlpfmKjBgYnzO/CMEewEmJtVs1Kz9ds/LTJUnljW36YF+t3t9fqw/21ajh6DZ6H+6vk9QVYscOcmhaXpKmDknSmTkJSj46rzPctXs6VeX8z7ZyVU632jyf3RA6LtKqzPhIZR5d0C4l1k5wBxASUuO6g/0wo0sJCV1D8KvV6fMrOzFKk7ITjC4JwKecMyxFgxKidKSxTf/aXqGvnjnY6JJCAsEewGnJSojSN6Zm6xtTs+Xz+bWv2qUNh+q1sbheGw/Vq6KpXVtKG7WltFFPrz0oSUqPjVCew6KhiREanhShYYkRirMHb/h+SkqKcnKCO6y0urlduyua9c5ul5Iv+6neKo+Qq+TgZ9pZTCalxnWtQp/liFSGI1JxkaxBACA0pcV3fdFqzxhhcCWhYccRp440tslqNmlWPvPqgVBkNpv0zanZenjVXr28qYxgfxTBHkDQmM0mjcyI08iMOH37rFz5/X6VNbRp46GukL+huF4Ha1pU5fKoyuXRR0faA6/1NtfJU1cmT12JPHWl8tSWylNfKl9Lk/SZweyfLyo6WruLik463Hs6fSpraNOhuhYdrm3RobpW7a92aXelU7WujkC72DEXyOXt+rMjKkIZ8V0BPiM+UilxNlnNrDEAIDykx3UNM49IG6JO38n9ru1vWr3SB/u7huCfPSyZhWGBEPbVMwfp4VV7tf5gnUrrW5WdFG10SYYj2APoNSaTSdlJ0cpOig58m/pBwSbN/vp1Ouuqn6gzOkUNHWa5vCZZ45JljUtW1JAJPc5hll9RVinK4leUxa9oqxRp8SvCJFnNfkWYFfizxSTVlh/WK4/fpQNlVbI50tTm6VRbR6faPJ1q93SqtaNTdS63aprdqnW5VXP0z1VOt440th33g63JJOWlxCjD3qk3VjytS+d/S2NHjRww6wcA6J8SoiNkNfnljYhUmdOrqUYXZKCP663q6PQpIz5SExiCD4S0wYnROntYstYdqNM/Pj6iH89i1BHBHsAxFRUV9cp5D+/fI3fpDk3OSdTgEaMkSW5vpxpaPKpv6VB9a0fXP1s65GzzyCeTWrxSi/dEh0OOUPaSF3Tda9XSa6tPur6oCItyk6M1JDlGuSnRykuO0ajMeI1Mj1OUzaLNmzfrLz/6qzKvXtCrob63/v5767wAwpPJZFKCza9at0n7Gzxf/IJ+Kjr/PFW2m2U2SbPz02RmCD4Q8uafOVjrDtTp75vL9KMLhw/4qTMEewA9OOtrJElXX311r17H5XIF/my3WpThsCjD0XPl4U6fXy1ur5rdXrnavXK5vWpu96i1o1MdnT51eLse7qP/7PT55fP75PP7ZTKZFWExKTLCoqgIi6JsXf+MjLAoKcam1Fi7UuN6PnKTopUaZzf0xmDE3z+AgS3R5letWzo4QIN9S4dPSbNukCRNG5LUbxZ4Bfq7S8ZlaOn/7dDhulZtOtygqUOSjC7JUAR7AD20uZySpEtv/KVGjg/+PupFG97Tm889qvb29i9sazGbFB8VofiTmOdYtm+nHl78VW3atEmTJ4ffPvCh9Pd/WtdhxAEQNhJtXVOQDgzQYP/STpcsMQmKs/o1ZYAHAyCcRNusmjcuUy8Xlulvm8oI9kYXACA0JWflavCIMUE/b1XJgaCf81jCfThWuP79M+IACD8JNp8k6VCjR95On6yWgbMA6P7qZr25v0WSNCHRy1akQJiZP3mwXi4s0xvbK3Tnl8cM6LWPCPYAgKDpLyMOgIEk1ir53K3qsEdrX7VL+ZnxRpfUJ/x+v+765y51+qXWfR8pPedMo0sCcJKmDUlSdlKUSuvb9NbOSl05aZDRJRmGYA8ACLpwHXEADEQmk9RRuV+RueO1raxxwAT71UXVen9fraxmqWHNn6RZBHsg3JjNJn110mA9unqf/lZYRrAHAADAwOWu2KPI3PHaUtqob07NMbqc4yopKVFtbe1pn8fT6dftb3VNHTo7uV0HGitO+5x9pTfWGklJSVFOTuj+ewc+z/wzu4L9hwdqVd7YpqyEKKNLMgTBHgAAYIBzl++RJH1c0mhsIZ+jpKREo/Lz1dbaetrnip82X4kXXCdvc51e+N33JYX+2h29uYZJVHS0dhcVEe4RlnKSozU9L0kFxfX6x8dHtPiC4UaXZAiCPQAAwADXcTTY76lqVnO7R3GRJ74bSV+pra1VW2urFt76oNJzhp3yedyd0sryCHn90lm5DrV+64awWLujt9YwqSo5oOX336La2lqCPcLW/MmDVVBcr78XlumH5w8L+0WUTwXBHgAAYIDrbGlQarRFNa2d2lbWpHOGpxhd0nGl5ww7rTU83t9XI6+/Uamxdp09cbg21+8LYnW9r7fWMAHC2bxxmfqf/9upg7Ut2lzSqMm5iUaX1OcGzn4mAAAAOK4zkrt66T8uaTC4kt7T3O7R1rImSdLZw5MHZK8e0B/F2q26ZGyGJOnvm8sMrsYYBHsAAADojGSbpNCeZ3+6Corr1enza1BClHKToo0uB0AQfW3yYEnSP7eWq93TaXA1fY9gDwAAgP/02Jc2yu/3G1xN8NW3dGhXedc89XPorQf6nbOGJmtQQpSa2716e1eV0eX0OYI9AAAANDQhQjaLWfUtHSqpP/2V50PN+oN18ksamhKjTMfA3A4L6M/MZpO+embXPvavDMDh+AR7AAAAKMJi0phB8ZKkjYf61zz7Kme79ld3bWc3Y1iywdUA6C1fmdQV7NfurVG1M7R3ugg2VsUH0C8VFRWF1XkBIBRMy0vSxyWN2lhcH5iv2h+sO1AnScrPiFNKrN3gagD0lqGpsTozJ0GbSxr1f1vKdf15Q40uqc8Q7AH0K876GknS1Vdf3avXcblcvXp+ADDC9Lwk/eG9g9pwqN7oUoKmpL5VJfWtMpu65uAC6N/mTx6szSWN+vvmMn3v3LwBs54GwR5Av9Lm6loY6dIbf6mR4ycH/fxFG97Tm889qvb2gTW8C8DAMDk3SSaTVFzbompnu9LiI40u6bT4/X6tO1ArSRo3yKH4qAiDKwLQ2y4bl6W7/rlLuyubtavCqTFZDqNL6hMEewD9UnJWrgaPGBP081aVHAj6OQEgVDiiIpSfEa9dFU5tOFSvy8ZnGV3SaTlQ06Iqp1sRFpOmDkkyuhwAfcARHaGL8tP1xvYK/b3wCMEeAAAAA8+0vKSuYF8c3sHe5/Nr/dG59ZOyExVj52MvEO5OdK2jCY52vSHp75sO6+LMNlnNxx+On5KSopycnCBVaBx+wwEAACBgel6Snl13SBuKw3uefVGlU/WtHYq0mnVmboLR5YS8YC4Oy0KzCLaTXkPJbNHgHz6rJiXq/AU/UNuBjcdtGhUdrd1FRWEf7gn2AAAACJia1zVkfXdlsxpaOpQYYzO4opPn7fTpo4NdX0xMHZIku9VicEWhqzcXnWWhWQTLqayhtLXBov3N0qRrbtdZKZ3HbFNVckDL779FtbW1BHsAAAD0Hymxdg1Pi9X+apc+OlinS8ZlGl3SSdt+pEkut1exdqvGDx4Y82tPVW8sOstCs+gtJ7OGkr3Zrf0bSlTZZlXKkBGKjOjfX/AR7AEAANDDzOEp2l/t0vv7a8Mu2Lu9ndp4qEFS17QCq8VscEXhIZiLzrLQLEJBSqxNybE21bk6tK/KpXH9/Es+ftMBAACgh/POSJEkrd1bI7/fb3A1J+fjkka1eTqVEB2h0ZnxRpcDwCAmk0mjM7p+BxRVOg2upvcR7AEAANDD9LxkRVhMKmto0+G6VqPLOWGtHV5tLunqrT97aLLMn7MSNoD+b2RGnEySKpra1dDaYXQ5vYpgDwAAgB5i7FadmZMoSXp/f63B1Zy4jYca5On0Ky2ua50AAANbjN2qnORoSdLuimaDq+ldBHsAAAB8xnlnpEqS3t9bY3AlJ8bZ5tH2siZJ0tnDkmUy0VsPQMr/xHD8cJtadDII9gAAAPiMc0d0zbNff6BO3k6fwdV8sY+K69Tp92twYpRykqKNLgdAiBiWGiOb1azmdq+ONLYZXU6vIdgDAADgM8ZkOZQYHaFmt1ebDjcYXc7nqnO5A8NszxmWQm89gACrxawzjk7N2VXRfxfRI9gDAABD3XfffZo6dari4uKUlpamK6+8Unv27OnRpr29XYsXL1ZycrJiY2M1f/58VVVV9WhTUlKiSy+9VNHR0UpLS9Mtt9wir9fbl2+lX7GYTbpgVJokadWuqi9obax1B+rkV1fPXIYj0uhyAISYUUd3yNhf7ZInDEYgnQqCPQAAMNR7772nxYsX66OPPtKqVavk8Xg0Z84ctbS0BNr85Cc/0T//+U+9/PLLeu+991ReXq6vfvWrgeOdnZ269NJL1dHRoXXr1um5557Ts88+q6VLlxrxlvqNOaMzJElv76oM2bmpRxradLC2RSZTV289AHxaliNSjqgIeTr9OlDtMrqcXmE1ugAAADCwrVy5ssfPzz77rNLS0lRYWKjzzjtPTU1N+tOf/qQVK1bowgsvlCT9+c9/Vn5+vj766COdddZZevvtt7Vr1y698847Sk9P18SJE3XPPffo1ltv1Z133imbzWbEWwt7552RIrvVrNL6Nu2pataojNDaF97v9+v9/V2L+43Ncigxhn/PAD7LZDJpVEacCorrVVTZHOjB709CuseeoXkAAAw8TU1dK5snJSVJkgoLC+XxeDR79uxAm1GjRiknJ0fr16+XJK1fv17jxo1Tenp6oM3cuXPldDq1c+fOY17H7XbL6XT2eKCnaJtVM4d39YKv2hl6w/H3VbtU5XQrwmLS9Lwko8sBEMLyj4b5kvpWNbd7DK4m+EI62DM0DwCAgcXn8+mmm27SOeeco7Fjx0qSKisrZbPZlJCQ0KNtenq6KisrA20+Geq7j3cfO5b77rtPDocj8MjOzg7yu+kfLhrd9fe4qii0gr3X59OH+2slSZNzExVjZyAqgONzREUoK6FrDY49lf1vT/uQDvYrV67Ud77zHY0ZM0YTJkzQs88+q5KSEhUWFkpSYGjeww8/rAsvvFCTJ0/Wn//8Z61bt04fffSRJAWG5r3wwguaOHGiLrnkEt1zzz164okn1NHRYeTbAwAAn7J48WLt2LFDL774Yq9f67bbblNTU1PgUVpa2uvXDEez8tNlMknbyppUWt9qdDkB28ua5Gz3KsZm0Zk5iUaXAyAMdPfaF1U0h+y6IacqpIP9pzE0DwCA/mvJkiV6/fXX9e6772rw4MGB5zMyMtTR0aHGxsYe7auqqpSRkRFo8+mpeN0/d7f5NLvdrvj4+B4PfFZqnF0zhiZLkl7bWm5wNV3cnk5tKK6XJJ01LFkRlrD6SAvAICPSYmUxm1Tf2qHqZrfR5QRV2PwWZGgeAAD9k9/v15IlS/SPf/xDa9asUV5eXo/jkydPVkREhFavXh14bs+ePSopKdGMGTMkSTNmzND27dtVXV0daLNq1SrFx8dr9OjRffNG+rErJw6SJP3j4yMh0cu1/mCd2r0+JcfYNLofLoIFoHfYrRYNS42RJBX1sz3twybYMzQPAID+afHixXrhhRe0YsUKxcXFqbKyUpWVlWpra5MkORwOLVq0SDfffLPeffddFRYW6rrrrtOMGTN01llnSZLmzJmj0aNH69vf/vb/b+/Ow6Oq7j+OfyaTfSOsWYQsAhpAQEgkRihSjYJS94VSwLBp1VA2RfBXEa1SlqotthSsKLigoBWpGyog4AMie9iNgAlhDbJkJyRkzu8PypSRLYFJJjd5v57nPg9zz7nnfs8Zcs/9zr1zRxs3btRXX32lZ555RmlpafLz8/Nk92qFHm0j5OvtpZ2HCrV1v2dPhnNLbdq099RdnF2vaiwvm82j8QCwltO342fkFMjh+c8p3cYSiT235gEAUHtNmzZNeXl56tatmyIjI53L3LlznXX++te/6je/+Y3uu+8+de3aVREREZo3b56z3G6367PPPpPdbldycrL69u2rhx56SH/605880aVaJ9TfR7e0OnXH4/wN+zwYiU3pR+0ykq5qEqzoBoEejAWAFUU3CFSQn10lZQ7tP157Phis0Yk9t+YBAFD7GWPOufTv399Zx9/fX1OnTtXRo0dVVFSkefPmnfUBfUxMjL744gsVFxfr559/1ksvvSRvb56U7i53dzh1O/5/Nu7XyXKHR2IIuubXOlLqJR+7Tb9q2dgjMQCwNi+bzfkVnqxCu4ejcZ8andhzax4AAEDNcONVjdUo2Fc/F5zQwm3V/9N3RaUO1e82UJKUFNdQwf58aAPg0pxO7HNKbLKH1I4PCWt0Ys+teQAAADWDr7eXeneKliS9tTKr2vf/3pYC2YPCFOJtdG2zsGrfP4DaIyzQV03DAiTZFNz2Zk+H4xY1+qPOijx19fSteVOnTj1vndO35gEAAODS/S4pWv9cukvf/3RUPxzMV3xE9TyHKH1Prr7aVSxJurbBSdm9as/3YgF4RpuoUO3NPa7gtily1IBf+7hcNfqKPQAAAGqOyHoB6t7m1EP03l65u1r2WVJWrlEfbpTDSEVbl6qJv/VPwAF4XosmwfKxGXmHRWjzoVJPh3PZSOwBAABQYQ8lx0qS5q3fq0MFJVW+vymLd2jHoUKF+Xvp6KLXqnx/AOoGb7uXmgWdehDo4p+KPRzN5SOxBwAAQIUlxTVQh+gwlZQ5NH3pT1W6rzVZR/Xasl2SpN8n1JOjpKBK9wegbokNPpXYf7+vRMeKrH3VnsQeAAAAFWaz2TTylqskSe+u2q2c/Kq5ap9bXKph72+Qw0j3dLhCSVf4V8l+ANRd9X2NSnN26aRDmp++z9PhXBYSewAAAFRKlxaNdF1sfZWedOgf3+x0e/vGGI3+aJP255UotmGgXrj7GrfvAwAkqWDj15KkuWv2VOjh7TUViT0AAAAqxWazacR/r9q/tzpbW/fnubX9fy7dpa+25sjHbtOrvTso2K9G/5ATAAsr3rZUPl7SDwcLtHmfe49l1YnEHgAAAJV2Q/NG6tkuUuUOo6fnbVa5wz1XuhZty9FLX2dIkp67s43aNQ1zS7sAcC6OE0W6vumpr/rMWbPHw9FcOhJ7AAAAXJJxd7RWiL+3Nu3N08wVmZfd3rrdxzR0zgYZI/W9Plp9kmLcECUAXFhKXKAk6dP0/TpeWu7haC4NiT0AAAAuSZMQfz19WytJ0qQvf9DarKOX3NbW/XkaMHO1ikvL9auWjTTujjbuChMALqhNE181axCgghMn9cXmA54O55KQ2AMAAOCS9e7UTLe3jVBZudGj767T/tzjlW5jbdZR9ZmxSvklJ5UQU1+v9UuQj53TVADVw8tmU6/EZpJOPUTPijhiAgAA4JLZbDb95f72io8I0eHCUv3u9e+1+0hRhbf/T/o+/W7GKuUWl6l9szC92f86BfrysDwA1ev+hGaye9m0Ouuofswp8HQ4lUZiDwAAgMsS5OetGamJalo/QFlHinXvP7/Td7sOX3CbY0WlGvr+Bg2bk67Skw6ltArXnIevV70An2qKGgD+J6Kev25pFS5Jevf73R6OpvJI7AEAAHDZmtYP1LzHb1CbqFAdKSrV715fpcfeXafvdh1W6UmHJOlkuUOb9+bpxc+2qfOkb/TJxv2ye9n0h5ta6LV+CQrwtXu4FwDqsr7Xn3pg57z1+1R04qSHo6kc7nMCAACAWzQJ8dfc3ydr4oLtem9VthZsOagFWw7Kz9tLgb52lZQ5dLzsf0+cbh0Zqj/f21bXNgvzXNAA8F83NG+oKxsF6afDRZqfvs9Sv8zBFXsAAAC4TbCft168u62+GPYr3dvxCjUM8tWJkw4dKy7T8bJyhfh56+b4Jpo14Dp9PrQLST2AGsPLy6bfJUVLkt5ZuVvGGA9HVHFcsQcAAIDbxUeE6pUHr5XDYbTnWLFKTzpk97IppmGQ7F42T4cHAOf0QEIzvfR1hn44WKD12ceUENPA0yFVCFfsAQAAUGW8/pvMtwwP0ZWNg0nqAdRo9QJ9dEe7KEmnrtpbBVfsAQAA4HbZ2dk6fPjCT8avjO3bt7utLQC4kH7JMfpw3V59sfmgxv7mhBoG+3k6pIsisQcAAIBbZWdnK75VKx0vLnZ724WFhW5vEwDO1K5pmNo1radNe/M0d+0ePd6thadDuigSewAAALjV4cOHdby4WH1G/0Xh0c3d0ub21cu04K0pKikpcUt7AHAhfa+P0VP/3qTZ32frkV9dKW97zf4WO4k9AAAAqkR4dHM1bdnGLW3lZO9ySzsAUBF3to/SpAU/aF/ucX259aB+89/v3ddUNftjBwAAAAAAqpm/j119rj/1O/ZvLM/0cDQXR2IPAAAAAMAv9Ls+Rr52L23IztX67GOeDueCSOwBAAAAAPiFxiF+uvPaU7fg1/Sr9iT2AAAAAACcw8DOcZKkL7cc1L7c4x6O5vxI7AEAAAAAOIfWUaFKvrKhyh1Gb32X5elwzovEHgAAAACA8xjU5dRV+/dXZ6voxEkPR3NuJPYAAAAAAJzHTfFNFNcoSAUlJ/XB2j2eDuec+B17AAAAAECdtX379ovWuSXarn8dlqYuzlAbv6Py9rKdt26jRo0UHR3tzhAvisQeAAAAFTqx9URbAFBV8o/+LEnq27fvRevavH11xe/f0GHV100DRqtoy+Lz1g0IDNQP27dXa3JPYg8AAFCHVebEtrIKCwvd3iYAuMvxwnxJUs/f/1FXt0u4aP2MfC9tyZVi7xquWx9Nk+0cF+1zsndp9qRROnz4MIk9AAAAqkdlT2wrYvvqZVrw1hSVlJS4pT0AqEoNo2LUtGWbi9ZrctKhHSsyVXjSoZJ60WoZHlIN0VUMiT0AAAAqfGJbETnZu9zSDgDUJL7eXrq2WZhWZR7V6qyjatEkWLZzXbb3AJ6KDwAAAABABbRvFiYfu02HC0uVdaTY0+E4kdgDAAAAAFABAT52tb2iniRpTdZRGWM8HNEpJPYAAAAAAFRQx+j6snvZdCCvRHuPHfd0OJJI7AEAAAAAqLAgP2+1iQyVJK386UiNuGpPYg8AAAAAQCVcF9dA3v+9ap95pMjT4ZDYAwAAAABQGcF+3mrfLEyS9N0uz1+1J7EHAAAAAKCSEmPqy8/bS0cKS5WRU+DRWEjsAQAAAACoJH8fuxJi6kuSVu46onKH567ak9gDAAAAAHAJrm0WpkBfu/JLTmrL/jyPxUFiDwAAAADAJfCxe6lTXANJ0urMozrp8EwcJPYAAAAAAFyia6LqqV6Aj4pLy/Vjvt0jMZDYAwAAAABwiexeNnVu3lCSlFHgJXto42qPgcQeAAAAAIDL0KJJsJqGBchhbKrfbUC175/EHgAAAACAy2Cz2dT1qsaSjIJaddXWn09U6/5J7AEAAAAAuEyNQ/wUF+zQif0/KNC7elNtEnsAAAAAANygfVi5Dr4zSnH1fap1vyT2AAAAAAC4gd1Lkky175fEHgAAAAAACyOxBwAAAADAwkjsAQAAAACwMBJ7AAAAAAAsjMQeAAAAAAALI7EHAAAAAMDCSOwBAAAAALAwEnsAAAAAACyMxB4AAAAAAAsjsQcAAAAAwMJI7AEAAAAAsDASewAAAAAALIzEHgAAAAAACyOxBwAAAADAwkjsAQAAAACwMBJ7AAAAAAAsjMQeAAAAAAALI7EHAAAAAMDCSOwBAAAAALAwEnsAAAAAACyMxB4AAAAAAAurU4n91KlTFRsbK39/fyUlJWn16tWeDgkAALgRcz0AoC6qM4n93LlzNXLkSI0bN07r169X+/bt1b17dx06dMjToQEAADdgrgcA1FV1JrF/5ZVX9PDDD2vAgAFq3bq1pk+frsDAQL355pueDg0AALgBcz0AoK6qE4l9aWmp1q1bp5SUFOc6Ly8vpaSkaOXKlR6MDAAAuANzPQCgLvP2dADV4fDhwyovL1d4eLjL+vDwcP3www9n1T9x4oROnDjhfJ2XlydJys/Pd0s8hYWFkqS9O7bqxPFit7R5ppzsXZKkg1k/aldQIO3TPu3TPu3XkPZ/3psp6dQ8cLlzyuntjTGXHVdtUNm5Xqra+b4q5vqq+v9ZFe1apc2qatcqbVZVu1Zps6ratUqbVdWuVdqsqnY9NtebOmDfvn1Gkvnuu+9c1o8aNcp06tTprPrjxo0zklhYWFhYWGr8smfPnuqaTmu0ys71xjDfs7CwsLBYY6nIXF8nrtg3atRIdrtdOTk5LutzcnIUERFxVv2nn35aI0eOdL52OBw6evSoGjZsKJvNdtnx5Ofnq1mzZtqzZ49CQ0Mvu72agD5ZR23sF32yjtrYL0/1yRijgoICRUVFVds+a7LKzvVS1c/351Mb/w6qEuNVeYxZ5TBelcN4Vd6ljlll5vo6kdj7+voqISFBixcv1t133y3p1OS9ePFiDRky5Kz6fn5+8vPzc1kXFhbm9rhCQ0Nr3R8DfbKO2tgv+mQdtbFfnuhTvXr1qnV/NVll53qp+ub786mNfwdVifGqPMaschivymG8Ku9Sxqyic32dSOwlaeTIkUpNTVViYqI6deqkv/3tbyoqKtKAAQM8HRoAAHAD5noAQF1VZxL7Xr166eeff9azzz6rgwcP6tprr9WXX3551kN2AACANTHXAwDqqjqT2EvSkCFDzns7XnXy8/PTuHHjzrr9z8rok3XUxn7RJ+uojf2qjX2yspoy118I/2cqh/GqPMaschivymG8Kq86xsxmDL+TAwAAAACAVXl5OgAAAAAAAHDpSOwBAAAAALAwEnsAAAAAACyMxB4AAAAAAAsjsa9mU6dOVWxsrPz9/ZWUlKTVq1d7OqRK+fbbb3XHHXcoKipKNptN8+fPdyk3xujZZ59VZGSkAgIClJKSoh07dngm2AqaMGGCrrvuOoWEhKhJkya6++67lZGR4VKnpKREaWlpatiwoYKDg3XfffcpJyfHQxFf3LRp09SuXTuFhoYqNDRUycnJWrBggbPcav05l4kTJ8pms2n48OHOdVbs13PPPSebzeayxMfHO8ut2CdJ2rdvn/r27auGDRsqICBAbdu21dq1a53lVjxWxMbGnvVe2Ww2paWlSbLue4WqURvnlupUW47xVa02HmurSnl5ucaOHau4uDgFBASoefPmeuGFF3Tmc8Tr+ni54zz/6NGj6tOnj0JDQxUWFqZBgwapsLCwGntRfS40XmVlZRo9erTatm2roKAgRUVF6aGHHtL+/ftd2nDneJHYV6O5c+dq5MiRGjdunNavX6/27dure/fuOnTokKdDq7CioiK1b99eU6dOPWf55MmT9eqrr2r69OlatWqVgoKC1L17d5WUlFRzpBW3bNkypaWl6fvvv9fChQtVVlamW2+9VUVFRc46I0aM0KeffqoPP/xQy5Yt0/79+3Xvvfd6MOoLa9q0qSZOnKh169Zp7dq1uummm3TXXXdp69atkqzXn19as2aNXnvtNbVr185lvVX71aZNGx04cMC5LF++3FlmxT4dO3ZMnTt3lo+PjxYsWKBt27bp5ZdfVv369Z11rHisWLNmjcv7tHDhQknSAw88IMma7xWqTm2cW6pLbTvGV5XaeqytKpMmTdK0adP0j3/8Q9u3b9ekSZM0efJk/f3vf3fWqevj5Y7z/D59+mjr1q1auHChPvvsM3377bd65JFHqqsL1epC41VcXKz169dr7NixWr9+vebNm6eMjAzdeeedLvXcOl4G1aZTp04mLS3N+bq8vNxERUWZCRMmeDCqSyfJfPzxx87XDofDREREmL/85S/Odbm5ucbPz8+8//77Hojw0hw6dMhIMsuWLTPGnOqDj4+P+fDDD511tm/fbiSZlStXeirMSqtfv76ZMWOG5ftTUFBgWrZsaRYuXGhuvPFGM2zYMGOMdd+ncePGmfbt25+zzKp9Gj16tOnSpct5y2vLsWLYsGGmefPmxuFwWPa9QvWprXOLu9W2Y3xVqivHWnfp2bOnGThwoMu6e++91/Tp08cYw3j90qWc52/bts1IMmvWrHHWWbBggbHZbGbfvn3VFrsn/HK8zmX16tVGktm9e7cxxv3jxRX7alJaWqp169YpJSXFuc7Ly0spKSlauXKlByNzn8zMTB08eNClj/Xq1VNSUpKl+piXlydJatCggSRp3bp1Kisrc+lXfHy8oqOjLdGv8vJyzZkzR0VFRUpOTrZ8f9LS0tSzZ0+X+CVrv087duxQVFSUrrzySvXp00fZ2dmSrNunTz75RImJiXrggQfUpEkTdejQQa+//rqzvDYcK0pLS/Xuu+9q4MCBstlsln2vUH1q29xSVWrjMb6q1IVjrTvdcMMNWrx4sX788UdJ0saNG7V8+XLddtttkhivi6nI+KxcuVJhYWFKTEx01klJSZGXl5dWrVpV7THXNHl5ebLZbAoLC5Pk/vHydleguLDDhw+rvLxc4eHhLuvDw8P1ww8/eCgq9zp48KAknbOPp8tqOofDoeHDh6tz58665pprJJ3ql6+vr/OP8LSa3q/NmzcrOTlZJSUlCg4O1scff6zWrVsrPT3dkv2RpDlz5mj9+vVas2bNWWVWfZ+SkpI0a9YsXX311Tpw4ICef/55/epXv9KWLVss26effvpJ06ZN08iRI/V///d/WrNmjYYOHSpfX1+lpqbWimPF/PnzlZubq/79+0uy7v8/VI/aNLdUpdp4jK9KdeFY605jxoxRfn6+4uPjZbfbVV5ervHjx6tPnz6Sasd5bFWqyPgcPHhQTZo0cSn39vZWgwYN6vwYlpSUaPTo0erdu7dCQ0MluX+8SOyBM6SlpWnLli0u33G2qquvvlrp6enKy8vTv//9b6WmpmrZsmWeDuuS7dmzR8OGDdPChQvl7+/v6XDc5vSVAklq166dkpKSFBMTow8++EABAQEejOzSORwOJSYm6s9//rMkqUOHDtqyZYumT5+u1NRUD0fnHm+88YZuu+02RUVFeToUWEBtmluqSm09xlelunCsdacPPvhAs2fP1nvvvac2bdooPT1dw4cPV1RUFOOFKlVWVqYHH3xQxhhNmzatyvbDrfjVpFGjRrLb7Wc9vTUnJ0cREREeisq9TvfDqn0cMmSIPvvsMy1ZskRNmzZ1ro+IiFBpaalyc3Nd6tf0fvn6+qpFixZKSEjQhAkT1L59e02ZMsWy/Vm3bp0OHTqkjh07ytvbW97e3lq2bJleffVVeXt7Kzw83JL9+qWwsDBdddVV2rlzp2Xfq8jISLVu3dplXatWrZxfMbD6sWL37t1atGiRBg8e7Fxn1fcKVa+2zS1Vpa4c492pth9r3W3UqFEaM2aMfvvb36pt27bq16+fRowYoQkTJkhivC6mIuMTERFx1kPBT548qaNHj9bZMTyd1O/evVsLFy50Xq2X3D9eJPbVxNfXVwkJCVq8eLFzncPh0OLFi5WcnOzByNwnLi5OERERLn3Mz8/XqlWranQfjTEaMmSIPv74Y33zzTeKi4tzKU9ISJCPj49LvzIyMpSdnV2j+/VLDodDJ06csGx/br75Zm3evFnp6enOJTExUX369HH+24r9+qXCwkLt2rVLkZGRln2vOnfufNbPev3444+KiYmRZN1jxWkzZ85UkyZN1LNnT+c6q75XqDp1ZW5xl7pyjHen2n6sdbfi4mJ5ebmmPna7XQ6HQxLjdTEVGZ/k5GTl5uZq3bp1zjrffPONHA6HkpKSqj1mTzud1O/YsUOLFi1Sw4YNXcrdPl6VftweLtmcOXOMn5+fmTVrltm2bZt55JFHTFhYmDl48KCnQ6uwgoICs2HDBrNhwwYjybzyyitmw4YNzqc7Tpw40YSFhZn//Oc/ZtOmTeauu+4ycXFx5vjx4x6O/Pwee+wxU69ePbN06VJz4MAB51JcXOys8+ijj5ro6GjzzTffmLVr15rk5GSTnJzswagvbMyYMWbZsmUmMzPTbNq0yYwZM8bYbDbz9ddfG2Os15/zOfOJycZYs19PPPGEWbp0qcnMzDQrVqwwKSkpplGjRubQoUPGGGv2afXq1cbb29uMHz/e7Nixw8yePdsEBgaad99911nHiscKY079mkl0dLQZPXr0WWVWfK9QdWrj3FLdasMxvirV5mNtVUhNTTVXXHGF+eyzz0xmZqaZN2+eadSokXnqqaecder6eLnjPL9Hjx6mQ4cOZtWqVWb58uWmZcuWpnfv3p7qUpW60HiVlpaaO++80zRt2tSkp6e7zAMnTpxwtuHO8SKxr2Z///vfTXR0tPH19TWdOnUy33//vadDqpQlS5YYSWctqampxphTP4UxduxYEx4ebvz8/MzNN99sMjIyPBv0RZyrP5LMzJkznXWOHz9uHn/8cVO/fn0TGBho7rnnHnPgwAHPBX0RAwcONDExMcbX19c0btzY3Hzzzc6k3hjr9ed8fnnSZ8V+9erVy0RGRhpfX19zxRVXmF69epmdO3c6y63YJ2OM+fTTT80111xj/Pz8THx8vPnXv/7lUm7FY4Uxxnz11VdG0jljtep7hapRG+eW6lYbjvFVrbYea6tCfn6+GTZsmImOjjb+/v7myiuvNH/84x9dkqy6Pl7uOM8/cuSI6d27twkODjahoaFmwIABpqCgwAO9qXoXGq/MzMzzzgNLlixxtuHO8bIZY0zlr/MDAAAAAICagO/YAwAAAABgYST2AAAAAABYGIk9AAAAAAAWRmIPAAAAAICFkdgDAAAAAGBhJPYAAAAAAFgYiT0AAAAAABZGYg8AAADUYTabTfPnzz9v+dKlS2Wz2ZSbm+vW/c6aNUthYWFubROoq0jsAZxXt27dNHz4cE+HAQBArdS/f3/ZbDY9+uijZ5WlpaXJZrOpf//+btvfc889p2uvvdZt7V3MkiVLdPvtt6thw4YKDAxU69at9cQTT2jfvn3VFgNQV5DYA6jxjDE6efKkp8MAAMDtmjVrpjlz5uj48ePOdSUlJXrvvfcUHR3twcguz2uvvaaUlBRFREToo48+0rZt2zR9+nTl5eXp5ZdfrtJ9l5WVVWn7QE1EYg/gnPr3769ly5ZpypQpstlsstlsysrK0pYtW3TbbbcpODhY4eHh6tevnw4fPuzcrlu3bho6dKieeuopNWjQQBEREXruueec5VlZWbLZbEpPT3euy83Nlc1m09KlSyX975a/BQsWKCEhQX5+flq+fLkcDocmTJiguLg4BQQEqH379vr3v/9dTSMCAID7dezYUc2aNdO8efOc6+bNm6fo6Gh16NDBue7EiRMaOnSomjRpIn9/f3Xp0kVr1qxxlp+eOxcvXqzExEQFBgbqhhtuUEZGhqRTt70///zz2rhxo3NenzVrlnP7w4cP65577lFgYKBatmypTz755JzxFhUVKTQ09Kz5d/78+QoKClJBQYH27t2roUOHaujQoXrzzTfVrVs3xcbGqmvXrpoxY4aeffZZl22/+uortWrVSsHBwerRo4cOHDjgLFuzZo1uueUWNWrUSPXq1dONN96o9evXu2xvs9k0bdo03XnnnQoKCtL48eMlSS+++KKaNGmikJAQDR48WGPGjDnrjoUZM2aoVatW8vf3V3x8vP75z386y0pLSzVkyBBFRkbK399fMTExmjBhwjnHBfA0EnsA5zRlyhQlJyfr4Ycf1oEDB3TgwAGFhITopptuUocOHbR27Vp9+eWXysnJ0YMPPuiy7VtvvaWgoCCtWrVKkydP1p/+9CctXLiw0jGMGTNGEydO1Pbt29WuXTtNmDBBb7/9tqZPn66tW7dqxIgR6tu3r5YtW+aubgMAUO0GDhyomTNnOl+/+eabGjBggEudp556Sh999JHeeustrV+/Xi1atFD37t119OhRl3p//OMf9fLLL2vt2rXy9vbWwIEDJUm9evXSE088oTZt2jjn9V69ejm3e/755/Xggw9q06ZNuv3229WnT5+z2pakoKAg/fa3v3WJV5Jmzpyp+++/XyEhIfrwww9VWlqqp5566pz9PfN79cXFxXrppZf0zjvv6Ntvv1V2draefPJJZ3lBQYFSU1O1fPlyff/992rZsqVuv/12FRQUuLT53HPP6Z577tHmzZs1cOBAzZ49W+PHj9ekSZO0bt06RUdHa9q0aS7bzJ49W88++6zGjx+v7du3689//rPGjh2rt956S5L06quv6pNPPtEHH3ygjIwMzZ49W7GxsefsE+BxBgDO48YbbzTDhg1zvn7hhRfMrbfe6lJnz549RpLJyMhwbtOlSxeXOtddd50ZPXq0McaYzMxMI8ls2LDBWX7s2DEjySxZssQYY8ySJUuMJDN//nxnnZKSEhMYGGi+++47l7YHDRpkevfufbldBQCg2qWmppq77rrLHDp0yPj5+ZmsrCyTlZVl/P39zc8//2zuuusuk5qaagoLC42Pj4+ZPXu2c9vS0lITFRVlJk+ebIz539y5aNEiZ53PP//cSDLHjx83xhgzbtw40759+7PikGSeeeYZ5+vCwkIjySxYsMCl7WPHjhljjFm1apWx2+1m//79xhhjcnJyjLe3t1m6dKkxxpjHHnvMhIaGXrT/M2fONJLMzp07neumTp1qwsPDz7tNeXm5CQkJMZ9++qlL/MOHD3epl5SUZNLS0lzWde7c2aX/zZs3N++9955LnRdeeMEkJycbY4z5wx/+YG666SbjcDgu2hfA07hiD6DCNm7cqCVLlig4ONi5xMfHS5J27drlrNeuXTuX7SIjI3Xo0KFK7y8xMdH57507d6q4uFi33HKLy/7ffvttl30DAGA1jRs3Vs+ePTVr1izNnDlTPXv2VKNGjZzlu3btUllZmTp37uxc5+Pjo06dOmn79u0ubZ05B0dGRkpShebgM7cLCgpSaGjoebfr1KmT2rRp47yy/e677yomJkZdu3aVdOrZODab7aL7lKTAwEA1b97cJeYz95uTk6OHH35YLVu2VL169RQaGqrCwkJlZ2e7tHPmOYMkZWRkqFOnTmfFfVpRUZF27dqlQYMGuZxXvPjii87ziv79+ys9PV1XX321hg4dqq+//rpCfQI8wdvTAQCwjsLCQt1xxx2aNGnSWWWnTx6kUycbZ7LZbHI4HJIkL69TnycaY5zl53vITVBQkMu+Jenzzz/XFVdc4VLPz8+vMt0AAKDGGThwoIYMGSJJmjp16iW3c+YcfDq5Pj0HV3S709teaLvBgwdr6tSpGjNmjGbOnKkBAwY493fVVVcpLy9PBw4ccDk/qOh+zzxHSE1N1ZEjRzRlyhTFxMTIz89PycnJKi0tddnuzHOGijh9XvH6668rKSnJpcxut0s69fyDzMxMLViwQIsWLdKDDz6olJQUnu+DGokr9gDOy9fXV+Xl5c7XHTt21NatWxUbG6sWLVq4LBWdUBs3bixJLg/GOfNBeufTunVr+fn5KTs7+6x9N2vWrHIdAwCghunRo4dKS0tVVlam7t27u5Q1b95cvr6+WrFihXNdWVmZ1qxZo9atW1d4H7+c1y9H3759tXv3br366qvatm2bUlNTnWX333+/fH19NXny5HNum5ubW+H9rFixQkOHDtXtt9+uNm3ayM/Pz+Whvedz9dVXuzxcUJLL6/DwcEVFRemnn34667wiLi7OWS80NFS9evXS66+/rrlz5+qjjz4657MHAE/jij2A84qNjdWqVauUlZWl4OBgpaWl6fXXX1fv3r2dT73fuXOn5syZoxkzZjg/4b6QgIAAXX/99Zo4caLi4uJ06NAhPfPMMxfdLiQkRE8++aRGjBghh8OhLl26KC8vTytWrFBoaKjLCQUAAFZjt9udt9X/cj4NCgrSY489plGjRqlBgwaKjo7W5MmTVVxcrEGDBlV4H7GxscrMzFR6erqaNm2qkJCQS77rrX79+rr33ns1atQo3XrrrWratKmzrFmzZvrrX/+qIUOGKD8/Xw899JBiY2O1d+9evf322woODq7wT961bNlS77zzjhITE5Wfn69Ro0YpICDgotv94Q9/0MMPP6zExETdcMMNmjt3rjZt2qQrr7zSWef555/X0KFDVa9ePfXo0UMnTpzQ2rVrdezYMY0cOVKvvPKKIiMj1aFDB3l5eenDDz9URESEy8P/gJqCK/YAzuvJJ5+U3W5X69at1bhxY5WWlmrFihUqLy/XrbfeqrZt22r48OEKCwtz3mJfEW+++aZOnjyphIQEDR8+XC+++GKFtnvhhRc0duxYTZgwQa1atVKPHj30+eefu3yyDgCAVYWGhio0NPScZRMnTtR9992nfv36qWPHjtq5c6e++uor1a9fv8Lt33ffferRo4d+/etfq3Hjxnr//fcvK95BgwaptLTU+eT9Mz3++OP6+uuvtW/fPt1zzz2Kj4/X4MGDFRoa6vLU+4t54403dOzYMXXs2FH9+vVz/uTfxfTp00dPP/20nnzySect9f3795e/v7+zzuDBgzVjxgzNnDlTbdu21Y033qhZs2Y5zytCQkI0efJkJSYm6rrrrlNWVpa++OKLSp3zANXFZs78EgsAAAAAVMA777yjESNGaP/+/fL19fV0OBd1yy23KCIiQu+8846nQwHcjlvxAQAAAFRYcXGxDhw4oIkTJ+r3v/99jUzqi4uLNX36dHXv3l12u13vv/++Fi1apIULF3o6NKBKcB8JAAAAgAqbPHmy4uPjFRERoaefftrT4ZyTzWbTF198oa5duyohIUGffvqpPvroI6WkpHg6NKBKcCs+AAAAAAAWxhV7AAAAAAAsjMQeAAAAAAALI7EHAAAAAMDCSOwBAAAAALAwEnsAAAAAACyMxB4AAAAAAAsjsQcAAAAAwMJI7AEAAAAAsDASewAAAAAALOz/AaQFqZc1hs2IAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1200x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(1,2, figsize=(12, 7))\n",
    "\n",
    "sns.histplot(df[\"tenure\"], ax=axes[0], kde =True)\n",
    "sns.histplot(df[\"MonthlyCharges\"], ax=axes[1], kde =True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tenure</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Churn</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>37.569965</td>\n",
       "      <td>61.265124</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>17.979133</td>\n",
       "      <td>74.441332</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          tenure  MonthlyCharges\n",
       "Churn                           \n",
       "0      37.569965       61.265124\n",
       "1      17.979133       74.441332"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['tenure','MonthlyCharges','Churn']].groupby('Churn').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* It is clear that people who have been a customer for a long time tend to stay with the company. \n",
    "* The **average tenure in months for people who left the company** is **20 months less than the average for people who stay.** \n",
    "\n",
    "* It seems like monthly charges also have an effect on churn rate. \n",
    "\n",
    "* Contract and tenure features may be correlated because customer with long term contract are likely to stay longer with the company. Let's figure out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tenure</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Contract</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Month-to-month</th>\n",
       "      <td>18.036645</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>One year</th>\n",
       "      <td>42.044807</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Two year</th>\n",
       "      <td>56.735103</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   tenure\n",
       "Contract                 \n",
       "Month-to-month  18.036645\n",
       "One year        42.044807\n",
       "Two year        56.735103"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Contract','tenure']].groupby('Contract').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* As expected, contract and tenure are highly correlated. \n",
    "* Customers with long contracts have been a customer for longer time than customers with short-term contracts. \n",
    "* I think contract will add little to no value to tenure feature so I will not use contract feature in the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After exploring the variables, I have decided not to use following variable because they add little or no informative power to the model:\n",
    "* 1) Customer ID\n",
    "* 2) Gender\n",
    "* 3) PhoneService\n",
    "* 4) Contract\n",
    "* 5) TotalCharges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>OnlineBackup</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   SeniorCitizen Partner Dependents  tenure     MultipleLines InternetService  \\\n",
       "0              0     Yes         No       1  No phone service             DSL   \n",
       "1              0      No         No      34                No             DSL   \n",
       "2              0      No         No       2                No             DSL   \n",
       "3              0      No         No      45  No phone service             DSL   \n",
       "4              0      No         No       2                No     Fiber optic   \n",
       "\n",
       "  OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV  \\\n",
       "0             No          Yes               No          No          No   \n",
       "1            Yes           No              Yes          No          No   \n",
       "2            Yes          Yes               No          No          No   \n",
       "3            Yes           No              Yes         Yes          No   \n",
       "4             No           No               No          No          No   \n",
       "\n",
       "  StreamingMovies PaperlessBilling              PaymentMethod  MonthlyCharges  \\\n",
       "0              No              Yes           Electronic check           29.85   \n",
       "1              No               No               Mailed check           56.95   \n",
       "2              No              Yes               Mailed check           53.85   \n",
       "3              No               No  Bank transfer (automatic)           42.30   \n",
       "4              No              Yes           Electronic check           70.70   \n",
       "\n",
       "   Churn  \n",
       "0      0  \n",
       "1      0  \n",
       "2      1  \n",
       "3      0  \n",
       "4      1  "
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7043, 16)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Categorical features need to be converted to numbers so that they can be included in calculations done by a machine learning model. \n",
    "* The categorical variables in our data set are not ordinal (i.e. there is no order in them). \n",
    "* For example, \"DSL\" internet service is not superior to \"Fiber optic\" internet service. \n",
    "* An example for an ordinal categorical variable would be ratings from 1 to 5 or a variable with categories \"bad\", \"average\" and \"good\". \n",
    "\n",
    "When we encode the categorical variables, a number will be assigned to each category. The category with higher numbers will be considered more important or effect the model more. Therefore, we need to do encode the variables in a way that each category will be represented by a column and the value in that column will be 0 or 1.\n",
    "\n",
    "We also need to scale continuous variables. Otherwise, variables with higher values will be given more importance which effects the accuracy of the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_features = ['SeniorCitizen', 'Partner', 'Dependents',\n",
    "        'MultipleLines', 'InternetService', 'OnlineSecurity',\n",
    "       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',\n",
    "       'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']\n",
    "X = pd.get_dummies(df, columns=cat_features, drop_first=True)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# pd.get_dummies(df, columns=cat_features, drop_first=True): \n",
    "# This line uses the pd.get_dummies() function from pandas \n",
    "# To perform one-hot encoding on the specified categorical features in df."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = MinMaxScaler()\n",
    "a = sc.fit_transform(df[['tenure']])\n",
    "b = sc.fit_transform(df[['MonthlyCharges']])\n",
    "\n",
    "# Min-max scaling is a preprocessing technique \n",
    "# That transforms numerical features to a specific range \n",
    "# (usually between 0 and 1) to ensure that they all have the same scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "X['tenure'] = a\n",
    "X['MonthlyCharges'] = b\n",
    "\n",
    "\n",
    "# Here we are updating the DataFrame X \n",
    "# With the scaled values of 'tenure' and 'MonthlyCharges'. \n",
    "# You're assigning the scaled values stored \n",
    "# In the variables a and b to the corresponding columns in X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7043, 26)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Resampling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* As we briefly discussed in the beginning, target variables with imbalanced class distribution is not desired for machine learning models. \n",
    "* I will use upsampling which means increasing the number of samples of the class with less samples by randomly selecting rows from it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Class Distribution Before Resampling')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHHCAYAAABeLEexAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA6WklEQVR4nO3deVxUZf//8fegMiI44MIiiYji7b4klZKmqSQZttxqplmSW6VYKaVmi5rd5Z1m7ktlRYvelVmamqjhVklpGGWW3tpNYhlgGqCogHB+f/Tl/JxwQQQHOq/n4zGPh+c611znc4aZ5t051zljMwzDEAAAgIW5uboAAAAAVyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQ4W+nYcOGuu+++1xdxmWbMmWKbDbbFdnWjTfeqBtvvNFc3rJli2w2mz744IMrsv377rtPDRs2vCLbulLi4+PVrl07Va9eXTabTZmZma4uCWf563v+559/ls1mU1xcnMtqgmsRiFBp/PTTT3rggQfUqFEjVa9eXQ6HQ506ddKcOXN06tQpV5d3QXFxcbLZbOajevXqCgwMVGRkpObOnavjx4+XyXYOHz6sKVOmKDk5uUzGK0sVsbaiL8GzHw6HQ+3atdP8+fNVUFBQqnGPHj2q/v37y8PDQwsWLNDbb78tT0/PMq7+0hWF7KJHtWrV1LBhQz388MMENlheVVcXAJTE2rVrdeedd8put2vw4MFq1aqV8vLy9Pnnn2vcuHHas2ePXnnlFVeXeVFTp05VSEiI8vPzlZaWpi1btmjMmDF66aWX9PHHH6tNmzZm36eeekqPP/74JY1/+PBhPfPMM2rYsKHatWtX4udt2LDhkrZTGheq7dVXX1VhYWG513A+AwcO1C233CJJysrK0ieffKKHHnpIBw8e1IwZMy55vJ07d+r48eN69tlnFRERUdblXrZFixbJy8tLOTk5SkhI0Lx587Rr1y59/vnnri7NZYKDg3Xq1ClVq1bN1aXARQhEqPBSUlI0YMAABQcHa9OmTapXr565LiYmRgcOHNDatWtdWGHJ9erVS9dcc425PHHiRG3atEm9e/fWbbfdph9//FEeHh6SpKpVq6pq1fL9iJ48eVI1atSQu7t7uW7nYlz9JdS+fXvdc8895vKoUaPUoUMHLVu2rFSBKCMjQ5Lk4+NTViUqJyenzI4y9evXT3Xr1pUkPfDAAxowYIDee+897dixQ9ddd12ZbKOyKTpyC+vilBkqvOnTp+vEiRN67bXXnMJQkdDQUD3yyCPnff6xY8f02GOPqXXr1vLy8pLD4VCvXr307bffFus7b948tWzZUjVq1FCtWrV0zTXXaNmyZeb648ePa8yYMWrYsKHsdrv8/Px00003adeuXaXev+7du+vpp5/WwYMH9c4775jt55pDtHHjRnXu3Fk+Pj7y8vJS06ZN9cQTT0j6c97PtddeK0kaMmSIeVqkaE7EjTfeqFatWikpKUldunRRjRo1zOf+dT5FkYKCAj3xxBMKCAiQp6enbrvtNh06dMipz/nmbJ095sVqO9ccopycHD366KMKCgqS3W5X06ZN9eKLL8owDKd+NptNo0eP1sqVK9WqVSvZ7Xa1bNlS8fHx537BS8Bms8nf3/+cgXTdunW64YYb5OnpqZo1ayoqKkp79uxx2u/o6GhJ0rXXXiubzeb0+ixfvlxhYWHy8PBQ3bp1dc899+jXX3912sZ9990nLy8v/fTTT7rllltUs2ZNDRo0SJJUWFio2bNnq2XLlqpevbr8/f31wAMP6I8//ij1/t5www2S/jwtfbavvvpKN998s7y9vVWjRg117dpVX3zxhVOfknwmPvvsM915551q0KCB7Ha7goKCNHbs2GKnuov2OzU1Vb1795aXl5euuuoqLViwQJK0e/dude/eXZ6engoODnb6bEr//9T0tm3b9MADD6hOnTpyOBwaPHjwRV+fc80hKqrn119/1R133CEvLy/5+vrqscceK3Y69ejRo7r33nvlcDjk4+Oj6Ohoffvtt8xLqkQ4QoQKb/Xq1WrUqJGuv/76Uj3/f//7n1auXKk777xTISEhSk9P18svv6yuXbvqhx9+UGBgoKQ/T9s8/PDD6tevnx555BGdPn1a3333nb766ivdfffdkqQHH3xQH3zwgUaPHq0WLVro6NGj+vzzz/Xjjz+qffv2pd7He++9V0888YQ2bNigESNGnLPPnj171Lt3b7Vp00ZTp06V3W7XgQMHzC+o5s2ba+rUqZo0aZLuv/9+80vu7Nft6NGj6tWrlwYMGKB77rlH/v7+F6zrueeek81m04QJE5SRkaHZs2crIiJCycnJ5pGskihJbWczDEO33XabNm/e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WbNG8fHxWrFihRYuXKhJkybpmWeekST1799fN9xwgz766CNt2LBBM2bM0AsvvKAPP/xQvXr1KnWNv/zyi7Kysi74H1APDw9t27ZNmzdv1tq1axUfH6/33ntP3bt314YNG1SlSpWLbudS5v2U1PnmZBQUFJSoprJwvu0Yf5mAfSl69Oih+fPna9u2bWrdurX5Xnn77bcVEBBQrH9ZXxF49pHNIoWFhfLz89PSpUvP+Zy/fnGfT5cuXcyrzG699Va1bt1agwYNUlJSktzc3Mx9nTFjxnlv3+Dl5SXp4p+JgoIC3XTTTTp27JgmTJigZs2aydPTU7/++qvuu+++Ykc7z/e3LI+/cUlcqfcwXI9AhAqvd+/eeuWVV5SYmKjw8PBLfv4HH3ygbt266bXXXnNqz8zMNL8Uinh6euquu+7SXXfdpby8PPXp00fPPfecJk6caJ7mqFevnkaNGqVRo0YpIyND7du313PPPXdZgejtt9+WJEVGRl6wn5ubm3r06KEePXropZde0vPPP68nn3xSmzdvVkRERJnf2broSEERwzB04MABp/sl1apV65w39Tt48KDTUYJLqS04OFiffvqpjh8/7nSUaO/eveb68nbmzBlJfx65k2SekvPz8yvVvYWKat63b1+xo0z79u0r0T41btxYn376qTp16lRm4dbLy0uTJ0/WkCFD9P7772vAgAHmvjocjhLt64U+E7t379Z///tfvfnmmxo8eLD5nI0bN5ZJ/eeyf/9+devWzVw+ceKEfvvtN/NeU2UtODhYmzdvNm9jUeTAgQPlsj2UD+YQocIbP368PD09NXz4cKWnpxdb/9NPP2nOnDnnfX6VKlWK/V/k8uXLi13qfPToUadld3d3tWjRQoZhKD8/XwUFBcUO7/v5+SkwMFC5ubmXulumTZs26dlnn1VISIh5afW5HDt2rFhb0f+9F22/6D41ZXXX4bfeesvpdOUHH3yg3377zSn8NW7cWF9++aXy8vLMtjVr1hS7PP9SarvllltUUFCg+fPnO7XPmjVLNpvtssJnSa1evVqS1LZtW0l/hlWHw6Hnn39e+fn5xfofOXLkguNdc8018vPz0+LFi53eL+vWrdOPP/6oqKioi9bUv39/FRQU6Nlnny227syZM6X+uw8aNEj169fXCy+8IEkKCwtT48aN9eKLL5qB8GxF+1qSz0TREZazP4OGYVzwM3u5XnnlFae/0aJFi3TmzJlye99ERkYqPz9fr776qtlWWFho3i4AlQNHiFDhNW7cWMuWLdNdd92l5s2bO92pevv27Vq+fPkFf7usd+/emjp1qoYMGaLrr79eu3fv1tKlS4vN++nZs6cCAgLUqVMn+fv768cff9T8+fMVFRWlmjVrKjMzU/Xr11e/fv3Utm1beXl56dNPP9XOnTs1c+bMEu3LunXrtHfvXp05c0bp6enatGmTNm7cqODgYH388ccXvDHc1KlTtW3bNkVFRSk4OFgZGRlauHCh6tevr86dO5uvlY+PjxYvXqyaNWvK09NTHTp0KHZZcknVrl1bnTt31pAhQ5Senq7Zs2crNDTUafLo8OHD9cEHH+jmm29W//799dNPP+mdd95xmuR8qbXdeuut6tatm5588kn9/PPPatu2rTZs2KBVq1ZpzJgxxca+XLt27TLvAXX8+HElJCRoxYoVuv7669WzZ09Jfx4tWbRoke699161b99eAwYMkK+vr1JTU7V27Vp16tSpWIA7W7Vq1fTCCy9oyJAh6tq1qwYOHGhedt+wYUONHTv2onV27dpVDzzwgKZNm6bk5GT17NlT1apV0/79+7V8+XLNmTNH/fr1u+T9r1atmh555BGNGzdO8fHxuvnmm7VkyRL16tVLLVu21JAhQ3TVVVfp119/1ebNm+VwOLR69WodP378op+JZs2aqXHjxnrsscf066+/yuFwaMWKFZd136SLycvLU48ePdS/f3/t27dPCxcuVOfOnXXbbbeVy/buuOMOXXfddXr00Ud14MABNWvWTB9//LH5PzFX6jcJcZlcdn0bcIn++9//GiNGjDAaNmxouLu7GzVr1jQ6depkzJs3zzh9+rTZ71yX3T/66KNGvXr1DA8PD6NTp05GYmJisctuX375ZaNLly5GnTp1DLvdbjRu3NgYN26ckZWVZRiGYeTm5hrjxo0z2rZta9SsWdPw9PQ02rZtayxcuPCitRddDlz0cHd3NwICAoybbrrJmDNnjtOl7UX+etl9QkKCcfvttxuBgYGGu7u7ERgYaAwcOND473//6/S8VatWGS1atDCqVq3qdMlv165dz3tbgfNddv+f//zHmDhxouHn52d4eHgYUVFRTpdJF5k5c6Zx1VVXGXa73ejUqZPx9ddfFxvzQrX99bJ7w/jzMvexY8ca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sjlENWvWVKtWrZzaPD09VadOHbN92LBhio2NVe3ateVwOPTQQw8pPDxcHTt2lCT17NlTLVq00L333qvp06crLS1NTz31lGJiYmS32yVJDz74oObPn6/x48dr6NCh2rRpk95//32tXbv2yu4wAACosFw6qfpiZs2aJTc3N/Xt21e5ubmKjIzUwoULzfVVqlTRmjVrNHLkSIWHh8vT01PR0dGaOnWq2SckJERr167V2LFjNWfOHNWvX19LlixRZGSkK3YJAABUQDbDMAxXF1HRZWdny9vbW1lZWeV6+ixs3FvlNjZQmSXNGOzqEgBUQpfy/e3y+xABAAC4GoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYHoEIAABYnksD0aJFi9SmTRs5HA45HA6Fh4dr3bp15vrTp08rJiZGderUkZeXl/r27av09HSnMVJTUxUVFaUaNWrIz89P48aN05kzZ5z6bNmyRe3bt5fdbldoaKji4uKuxO4BAIBKwqWBqH79+vr3v/+tpKQkff311+revbtuv/127dmzR5I0duxYrV69WsuXL9fWrVt1+PBh9enTx3x+QUGBoqKilJeXp+3bt+vNN99UXFycJk2aZPZJSUlRVFSUunXrpuTkZI0ZM0bDhw/X+vXrr/j+AgCAislmGIbh6iLOVrt2bc2YMUP9+vWTr6+vli1bpn79+kmS9u7dq+bNmysxMVEdO3bUunXr1Lt3bx0+fFj+/v6SpMWLF2vChAk6cuSI3N3dNWHCBK1du1bff/+9uY0BAwYoMzNT8fHxJaop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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x ='Churn', data=df).set_title('Class Distribution Before Resampling')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_no = X[X.Churn == 0]\n",
    "X_yes = X[X.Churn == 1]\n",
    "\n",
    "\n",
    "\n",
    "# We are creating two subsets of the DataFrame X\n",
    "# Based on the values in the 'Churn' column. \n",
    "# The subsets X_no and X_yes contain rows \n",
    "# where the 'Churn' column is equal to 0 (not churned) and 1 (churned).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5174 1869\n"
     ]
    }
   ],
   "source": [
    "print(len(X_no),len(X_yes))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5174\n"
     ]
    }
   ],
   "source": [
    "X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)\n",
    "print(len(X_yes_upsampled))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tenure</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>Churn</th>\n",
       "      <th>SeniorCitizen_1</th>\n",
       "      <th>Partner_Yes</th>\n",
       "      <th>Dependents_Yes</th>\n",
       "      <th>MultipleLines_No phone service</th>\n",
       "      <th>MultipleLines_Yes</th>\n",
       "      <th>InternetService_Fiber optic</th>\n",
       "      <th>InternetService_No</th>\n",
       "      <th>...</th>\n",
       "      <th>TechSupport_No internet service</th>\n",
       "      <th>TechSupport_Yes</th>\n",
       "      <th>StreamingTV_No internet service</th>\n",
       "      <th>StreamingTV_Yes</th>\n",
       "      <th>StreamingMovies_No internet service</th>\n",
       "      <th>StreamingMovies_Yes</th>\n",
       "      <th>PaperlessBilling_Yes</th>\n",
       "      <th>PaymentMethod_Credit card (automatic)</th>\n",
       "      <th>PaymentMethod_Electronic check</th>\n",
       "      <th>PaymentMethod_Mailed check</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.013889</td>\n",
       "      <td>0.115423</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.472222</td>\n",
       "      <td>0.385075</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.625000</td>\n",
       "      <td>0.239303</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.305556</td>\n",
       "      <td>0.704975</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.138889</td>\n",
       "      <td>0.114428</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     tenure  MonthlyCharges  Churn  SeniorCitizen_1  Partner_Yes  \\\n",
       "0  0.013889        0.115423      0            False         True   \n",
       "1  0.472222        0.385075      0            False        False   \n",
       "3  0.625000        0.239303      0            False        False   \n",
       "6  0.305556        0.704975      0            False        False   \n",
       "7  0.138889        0.114428      0            False        False   \n",
       "\n",
       "   Dependents_Yes  MultipleLines_No phone service  MultipleLines_Yes  \\\n",
       "0           False                            True              False   \n",
       "1           False                           False              False   \n",
       "3           False                            True              False   \n",
       "6            True                           False               True   \n",
       "7           False                            True              False   \n",
       "\n",
       "   InternetService_Fiber optic  InternetService_No  ...  \\\n",
       "0                        False               False  ...   \n",
       "1                        False               False  ...   \n",
       "3                        False               False  ...   \n",
       "6                         True               False  ...   \n",
       "7                        False               False  ...   \n",
       "\n",
       "   TechSupport_No internet service  TechSupport_Yes  \\\n",
       "0                            False            False   \n",
       "1                            False            False   \n",
       "3                            False             True   \n",
       "6                            False            False   \n",
       "7                            False            False   \n",
       "\n",
       "   StreamingTV_No internet service  StreamingTV_Yes  \\\n",
       "0                            False            False   \n",
       "1                            False            False   \n",
       "3                            False            False   \n",
       "6                            False             True   \n",
       "7                            False            False   \n",
       "\n",
       "   StreamingMovies_No internet service  StreamingMovies_Yes  \\\n",
       "0                                False                False   \n",
       "1                                False                False   \n",
       "3                                False                False   \n",
       "6                                False                False   \n",
       "7                                False                False   \n",
       "\n",
       "   PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  \\\n",
       "0                  True                                  False   \n",
       "1                 False                                  False   \n",
       "3                 False                                  False   \n",
       "6                  True                                   True   \n",
       "7                 False                                  False   \n",
       "\n",
       "   PaymentMethod_Electronic check  PaymentMethod_Mailed check  \n",
       "0                            True                       False  \n",
       "1                           False                        True  \n",
       "3                           False                       False  \n",
       "6                           False                       False  \n",
       "7                           False                        True  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_yes_upsampled.head()\n",
    "X_no.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_upsampled= pd.concat([X_no,X_yes_upsampled]).reset_index(drop=True)\n",
    "# adding the upsampled yes cases to the no cases\n",
    "# to balance the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tenure</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>Churn</th>\n",
       "      <th>SeniorCitizen_1</th>\n",
       "      <th>Partner_Yes</th>\n",
       "      <th>Dependents_Yes</th>\n",
       "      <th>MultipleLines_No phone service</th>\n",
       "      <th>MultipleLines_Yes</th>\n",
       "      <th>InternetService_Fiber optic</th>\n",
       "      <th>InternetService_No</th>\n",
       "      <th>...</th>\n",
       "      <th>TechSupport_No internet service</th>\n",
       "      <th>TechSupport_Yes</th>\n",
       "      <th>StreamingTV_No internet service</th>\n",
       "      <th>StreamingTV_Yes</th>\n",
       "      <th>StreamingMovies_No internet service</th>\n",
       "      <th>StreamingMovies_Yes</th>\n",
       "      <th>PaperlessBilling_Yes</th>\n",
       "      <th>PaymentMethod_Credit card (automatic)</th>\n",
       "      <th>PaymentMethod_Electronic check</th>\n",
       "      <th>PaymentMethod_Mailed check</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.013889</td>\n",
       "      <td>0.115423</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.472222</td>\n",
       "      <td>0.385075</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.625000</td>\n",
       "      <td>0.239303</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.305556</td>\n",
       "      <td>0.704975</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.138889</td>\n",
       "      <td>0.114428</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10343</th>\n",
       "      <td>0.708333</td>\n",
       "      <td>0.408458</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10344</th>\n",
       "      <td>0.041667</td>\n",
       "      <td>0.518905</td>\n",
       "      <td>1</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10345</th>\n",
       "      <td>0.833333</td>\n",
       "      <td>0.771144</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10346</th>\n",
       "      <td>0.027778</td>\n",
       "      <td>0.309950</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10347</th>\n",
       "      <td>0.013889</td>\n",
       "      <td>0.516418</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10348 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         tenure  MonthlyCharges  Churn  SeniorCitizen_1  Partner_Yes  \\\n",
       "0      0.013889        0.115423      0            False         True   \n",
       "1      0.472222        0.385075      0            False        False   \n",
       "2      0.625000        0.239303      0            False        False   \n",
       "3      0.305556        0.704975      0            False        False   \n",
       "4      0.138889        0.114428      0            False        False   \n",
       "...         ...             ...    ...              ...          ...   \n",
       "10343  0.708333        0.408458      1            False         True   \n",
       "10344  0.041667        0.518905      1             True        False   \n",
       "10345  0.833333        0.771144      1            False         True   \n",
       "10346  0.027778        0.309950      1            False        False   \n",
       "10347  0.013889        0.516418      1            False        False   \n",
       "\n",
       "       Dependents_Yes  MultipleLines_No phone service  MultipleLines_Yes  \\\n",
       "0               False                            True              False   \n",
       "1               False                           False              False   \n",
       "2               False                            True              False   \n",
       "3                True                           False               True   \n",
       "4               False                            True              False   \n",
       "...               ...                             ...                ...   \n",
       "10343            True                            True              False   \n",
       "10344           False                           False              False   \n",
       "10345           False                           False               True   \n",
       "10346           False                           False               True   \n",
       "10347           False                           False              False   \n",
       "\n",
       "       InternetService_Fiber optic  InternetService_No  ...  \\\n",
       "0                            False               False  ...   \n",
       "1                            False               False  ...   \n",
       "2                            False               False  ...   \n",
       "3                             True               False  ...   \n",
       "4                            False               False  ...   \n",
       "...                            ...                 ...  ...   \n",
       "10343                        False               False  ...   \n",
       "10344                         True               False  ...   \n",
       "10345                         True               False  ...   \n",
       "10346                        False               False  ...   \n",
       "10347                         True               False  ...   \n",
       "\n",
       "       TechSupport_No internet service  TechSupport_Yes  \\\n",
       "0                                False            False   \n",
       "1                                False            False   \n",
       "2                                False             True   \n",
       "3                                False            False   \n",
       "4                                False            False   \n",
       "...                                ...              ...   \n",
       "10343                            False            False   \n",
       "10344                            False            False   \n",
       "10345                            False            False   \n",
       "10346                            False            False   \n",
       "10347                            False            False   \n",
       "\n",
       "       StreamingTV_No internet service  StreamingTV_Yes  \\\n",
       "0                                False            False   \n",
       "1                                False            False   \n",
       "2                                False            False   \n",
       "3                                False             True   \n",
       "4                                False            False   \n",
       "...                                ...              ...   \n",
       "10343                            False             True   \n",
       "10344                            False            False   \n",
       "10345                            False             True   \n",
       "10346                            False            False   \n",
       "10347                            False            False   \n",
       "\n",
       "       StreamingMovies_No internet service  StreamingMovies_Yes  \\\n",
       "0                                    False                False   \n",
       "1                                    False                False   \n",
       "2                                    False                False   \n",
       "3                                    False                False   \n",
       "4                                    False                False   \n",
       "...                                    ...                  ...   \n",
       "10343                                False                 True   \n",
       "10344                                False                False   \n",
       "10345                                False                False   \n",
       "10346                                False                False   \n",
       "10347                                False                False   \n",
       "\n",
       "       PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  \\\n",
       "0                      True                                  False   \n",
       "1                     False                                  False   \n",
       "2                     False                                  False   \n",
       "3                      True                                   True   \n",
       "4                     False                                  False   \n",
       "...                     ...                                    ...   \n",
       "10343                 False                                  False   \n",
       "10344                  True                                  False   \n",
       "10345                  True                                  False   \n",
       "10346                  True                                  False   \n",
       "10347                  True                                  False   \n",
       "\n",
       "       PaymentMethod_Electronic check  PaymentMethod_Mailed check  \n",
       "0                                True                       False  \n",
       "1                               False                        True  \n",
       "2                               False                       False  \n",
       "3                               False                       False  \n",
       "4                               False                        True  \n",
       "...                               ...                         ...  \n",
       "10343                            True                       False  \n",
       "10344                            True                       False  \n",
       "10345                            True                       False  \n",
       "10346                            True                       False  \n",
       "10347                           False                        True  \n",
       "\n",
       "[10348 rows x 26 columns]"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_upsampled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Class Distribution After Resampling')"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHHCAYAAABeLEexAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA5xUlEQVR4nO3deVxU9f7H8feAMiA44MIiuSFY7nrF0slyiyTDlptamjfNpcywruJVo8zUFm+aa65lSV21UitLLZdwK8UyupSaetWraSlgKqAmoHB+f3Q5PydcERzovJ6Pxzwene/5zvd8zjDTvD3ne87YDMMwBAAAYGEe7i4AAADA3QhEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEsJzatWvr0UcfdXcZ12z06NGy2WzXZVvt2rVTu3btzOX169fLZrNpyZIl12X7jz76qGrXrn1dtnUttm7dqltvvVW+vr6y2WxKSUlxd0mWduDAAdlsNiUkJJht1/Nzg7KFQIQ/jX379mnAgAGqU6eOvL295XA41Lp1a02dOlVnzpxxd3mXlJCQIJvNZj68vb0VGhqq6OhoTZs2TSdPniyW7Rw+fFijR48ulV/Upbk2Sdq5c6f5t8nIyCi0/uzZs+rWrZuOHz+uyZMn61//+pdq1aqlmTNnunwhXw+1a9d2eT/5+vrqlltu0bvvvntd6wDKknLuLgAoDitWrFC3bt1kt9vVq1cvNWrUSLm5ufrqq680bNgw7dixQ2+88Ya7y7yssWPHKiwsTGfPnlVqaqrWr1+vwYMHa9KkSfr000/VpEkTs+/IkSP1zDPPXNX4hw8f1pgxY1S7dm01a9bsip+3evXqq9pOUVyqtjfffFP5+fklXsOlzJ8/XyEhITpx4oSWLFmi/v37u6zft2+ffvrpJ7355psu62bOnKmqVate96OSzZo109ChQyVJR44c0dy5c9W7d2/l5OToscceu661lCZF+dzAGghEKPP279+v7t27q1atWlq7dq2qVatmrouNjdXevXu1YsUKN1Z45Tp16qQWLVqYy/Hx8Vq7dq06d+6se++9Vzt37pSPj48kqVy5cipXrmQ/wr/99psqVKggLy+vEt3O5ZQvX96t2zcMQwsXLtTDDz+s/fv3a8GCBYUCUXp6uiQpICCgxOs5d+6c8vPzL/l3ueGGG/S3v/3NXH700UdVp04dTZ482dKB6Hp8blA2ccoMZd748eN16tQpvfXWWy5hqEBERIT+/ve/X/T5x48f1z/+8Q81btxYfn5+cjgc6tSpk77//vtCfV9//XU1bNhQFSpUUKVKldSiRQstXLjQXH/y5EkNHjxYtWvXlt1uV1BQkO6880599913Rd6/Dh066Pnnn9dPP/2k+fPnm+0XmguxZs0a3XbbbQoICJCfn59uuukmPfvss5J+n/dz8803S5L69Oljnk4pOJ3Trl07NWrUSMnJyWrTpo0qVKhgPvePc4gK5OXl6dlnn1VISIh8fX1177336tChQy59LjZn6/wxL1fbheYQnT59WkOHDlWNGjVkt9t100036bXXXpNhGC79bDabBg0apKVLl6pRo0ay2+1q2LChVq5ceeEX/AI2bdqkAwcOqHv37urevbs2btyon3/+2Vz/6KOPqm3btpKkbt26yWazqV27dqpdu7Z27NihDRs2mPt0/uuYkZGhwYMHm/sQERGhV1991eVoWME8mNdee01TpkxReHi47Ha7fvzxxyuuX5ICAwNVr1497du3z6U9Pz9fU6ZMUcOGDeXt7a3g4GANGDBAJ06ccOn37bffKjo6WlWrVpWPj4/CwsLUt29flz6vvfaabr31VlWpUkU+Pj6KjIy84Dyzgr/J4sWL1aBBA/n4+MjpdGrbtm2SpDlz5igiIkLe3t5q166dDhw44PL889+rt956q1nP7NmzL/s6XOhzczXvkfXr16tFixby9vZWeHi45syZw7ykPwliMsq8ZcuWqU6dOrr11luL9Pz//ve/Wrp0qbp166awsDClpaVpzpw5atu2rX788UeFhoZK+v20zdNPP62uXbvq73//u7Kzs/XDDz/o66+/1sMPPyxJeuKJJ7RkyRINGjRIDRo00LFjx/TVV19p586dat68eZH38ZFHHtGzzz6r1atXX/Rf9zt27FDnzp3VpEkTjR07Vna7XXv37tWmTZskSfXr19fYsWM1atQoPf7447r99tslyeV1O3bsmDp16qTu3bvrb3/7m4KDgy9Z18svvyybzaYRI0YoPT1dU6ZMUVRUlFJSUswjWVfiSmo7n2EYuvfee7Vu3Tr169dPzZo106pVqzRs2DD98ssvmjx5skv/r776Sh999JGefPJJVaxYUdOmTVOXLl108OBBValS5bL1LViwQOHh4br55pvVqFEjVahQQe+9956GDRsmSRowYIBuuOEGvfLKK3r66ad18803Kzg4WKdPn9ZTTz0lPz8/Pffcc5Jkvqa//fab2rZtq19++UUDBgxQzZo1tXnzZsXHx+vIkSOaMmWKSw3z5s1Tdna2Hn/8cdntdlWuXPmKX1/p96NKP//8sypVquTSPmDAACUkJKhPnz56+umntX//fk2fPl3//ve/tWnTJpUvX17p6enq2LGjAgMD9cwzzyggIEAHDhzQRx995DLW1KlTde+996pnz57Kzc3V+++/r27dumn58uWKiYlx6fvll1/q008/VWxsrCRp3Lhx6ty5s4YPH66ZM2fqySef1IkTJzR+/Hj17dtXa9eudXn+iRMndPfdd+vBBx9Ujx49tGjRIg0cOFBeXl6FgtqVuJL3yL///W/dddddqlatmsaMGaO8vDyNHTtWgYGBV709lEIGUIZlZmYakoz77rvvip9Tq1Yto3fv3uZydna2kZeX59Jn//79ht1uN8aOHWu23XfffUbDhg0vOba/v78RGxt7xbUUmDdvniHJ2Lp16yXH/stf/mIuv/DCC8b5H+HJkycbkoyjR49edIytW7cakox58+YVWte2bVtDkjF79uwLrmvbtq25vG7dOkOSccMNNxhZWVlm+6JFiwxJxtSpU822P77eFxvzUrX17t3bqFWrlrm8dOlSQ5Lx0ksvufTr2rWrYbPZjL1795ptkgwvLy+Xtu+//96QZLz++uuFtvVHubm5RpUqVYznnnvObHv44YeNpk2buvQreE0WL17s0t6wYUOX/Szw4osvGr6+vsZ//vMfl/ZnnnnG8PT0NA4ePGgYxu/vRUmGw+Ew0tPTL1uvYfz+mnfs2NE4evSocfToUWPbtm3GI488YkhyeX9++eWXhiRjwYIFLs9fuXKlS/vHH3982fenYRjGb7/95rKcm5trNGrUyOjQoYNLuyTDbrcb+/fvN9vmzJljSDJCQkJc3lPx8fGGJJe+Be/ViRMnmm05OTlGs2bNjKCgICM3N9cwjP9/7c5/T/3xc1NQz5W8R+655x6jQoUKxi+//GK27dmzxyhXrlyhMVH2cMoMZVpWVpYkqWLFikUew263y8Pj949CXl6ejh07Zp5uOv9UV0BAgH7++Wdt3br1omMFBATo66+/1uHDh4tcz8X4+fld8mqzgrkrn3zySZEnINvtdvXp0+eK+/fq1cvlte/atauqVaumzz77rEjbv1KfffaZPD099fTTT7u0Dx06VIZh6PPPP3dpj4qKUnh4uLncpEkTORwO/fe//73stj7//HMdO3ZMPXr0MNt69Oih77//Xjt27CjyPixevFi33367KlWqpF9//dV8REVFKS8vTxs3bnTp36VLl6s6ErF69WoFBgYqMDBQjRs31r/+9S/16dNHEyZMcKnB399fd955p0sNkZGR8vPz07p16yT9/3tr+fLlOnv27EW3ef5RwRMnTigzM1O33377BU8Z33HHHS6nQVu2bGnu5/nvqYL2P/6typUrpwEDBpjLXl5eGjBggNLT05WcnHy5l6eQy71H8vLy9MUXX+j+++83jxpLv5+S79Sp01VvD6UPgQhlmsPhkKRruiw9Pz9fkydPVt26dWW321W1alUFBgbqhx9+UGZmptlvxIgR8vPz0y233KK6desqNjbWPB1VYPz48dq+fbtq1KihW265RaNHj76iL90rcerUqUsGv4ceekitW7dW//79FRwcrO7du2vRokVXFY5uuOGGq5pAXbduXZdlm82miIiIQnM+ittPP/2k0NDQQq9H/fr1zfXnq1mzZqExKlWqVGiezIXMnz9fYWFh5inIvXv3Kjw8XBUqVNCCBQuKvA979uzRypUrzdBS8IiKipL0/5O0C4SFhV3V+C1bttSaNWu0cuVKvfbaawoICNCJEydc/r579uxRZmamgoKCCtVx6tQps4a2bduqS5cuGjNmjKpWrar77rtP8+bNU05Ojss2ly9frlatWsnb21uVK1dWYGCgZs2a5fI5KvDHv4m/v78kqUaNGhds/+PfKjQ0VL6+vi5tN954oyQV6f13ufdIenq6zpw5o4iIiEL9LtSGsoc5RCjTHA6HQkNDtX379iKP8corr+j5559X37599eKLL6py5cry8PDQ4MGDXcJE/fr1tXv3bi1fvlwrV67Uhx9+qJkzZ2rUqFEaM2aMJOnBBx/U7bffro8//lirV6/WhAkT9Oqrr+qjjz66pn9F/vzzz8rMzLzk/3h9fHy0ceNGrVu3TitWrNDKlSv1wQcfqEOHDlq9erU8PT0vu52rmfdzpS422TQvL++KaioOF9uO8YcJ2H+UlZWlZcuWKTs7u1D4k6SFCxea86iuVn5+vu68804N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AADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8ghEAADA8twaiDZu3Kh77rlHoaGhstlsWrp0qct6wzA0atQoVatWTT4+PoqKitKePXtc+hw/flw9e/aUw+FQQECA+vXrp1OnTrn0+eGHH3T77bfL29tbNWrU0Pjx40t61wAAQBni1kB0+vRpNW3aVDNmzLjg+vHjx2vatGmaPXu2vv76a/n6+io6OlrZ2dlmn549e2rHjh1as2aNli9fro0bN+rxxx8312dlZaljx46qVauWkpOTNWHCBI0ePVpvvPFGie8fAAAoG8q5c+OdOnVSp06dLrjOMAxNmTJFI0eO1H333SdJevfddxUcHKylS5eqe/fu2rlzp1auXKmtW7eqRYsWkqTXX39dd999t1577TWFhoZqwYIFys3N1dtvvy0vLy81bNhQKSkpmjRpkktwAgAA1lVq5xDt379fqampioqKMtv8/f3VsmVLJSUlSZKSkpIUEBBghiFJioqKkoeHh77++muzT5s2beTl5WX2iY6O1u7du3XixInrtDcAAKA0c+sRoktJTU2VJAUHB7u0BwcHm+tSU1MVFBTksr5cuXKqXLmyS5+wsLBCYxSsq1SpUqFt5+TkKCcnx1zOysq6xr0BAAClWak9QuRO48aNk7+/v/moUaOGu0sCAAAlqNQGopCQEElSWlqaS3taWpq5LiQkROnp6S7rz507p+PHj7v0udAY52/jj+Lj45WZmWk+Dh06dO07BAAASq1SG4jCwsIUEhKixMREsy0rK0tff/21nE6nJMnpdCojI0PJyclmn7Vr1yo/P18tW7Y0+2zcuFFn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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x ='Churn', data=X_upsampled).set_title('Class Distribution After Resampling')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ML model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We need to divide the data set into training and test subsets so that we are able to measure the performance of our model on new, previously unseen examples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tenure</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>Churn</th>\n",
       "      <th>SeniorCitizen_1</th>\n",
       "      <th>Partner_Yes</th>\n",
       "      <th>Dependents_Yes</th>\n",
       "      <th>MultipleLines_No phone service</th>\n",
       "      <th>MultipleLines_Yes</th>\n",
       "      <th>InternetService_Fiber optic</th>\n",
       "      <th>InternetService_No</th>\n",
       "      <th>...</th>\n",
       "      <th>TechSupport_No internet service</th>\n",
       "      <th>TechSupport_Yes</th>\n",
       "      <th>StreamingTV_No internet service</th>\n",
       "      <th>StreamingTV_Yes</th>\n",
       "      <th>StreamingMovies_No internet service</th>\n",
       "      <th>StreamingMovies_Yes</th>\n",
       "      <th>PaperlessBilling_Yes</th>\n",
       "      <th>PaymentMethod_Credit card (automatic)</th>\n",
       "      <th>PaymentMethod_Electronic check</th>\n",
       "      <th>PaymentMethod_Mailed check</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.013889</td>\n",
       "      <td>0.115423</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.472222</td>\n",
       "      <td>0.385075</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.625000</td>\n",
       "      <td>0.239303</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.305556</td>\n",
       "      <td>0.704975</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.138889</td>\n",
       "      <td>0.114428</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     tenure  MonthlyCharges  Churn  SeniorCitizen_1  Partner_Yes  \\\n",
       "0  0.013889        0.115423      0            False         True   \n",
       "1  0.472222        0.385075      0            False        False   \n",
       "2  0.625000        0.239303      0            False        False   \n",
       "3  0.305556        0.704975      0            False        False   \n",
       "4  0.138889        0.114428      0            False        False   \n",
       "\n",
       "   Dependents_Yes  MultipleLines_No phone service  MultipleLines_Yes  \\\n",
       "0           False                            True              False   \n",
       "1           False                           False              False   \n",
       "2           False                            True              False   \n",
       "3            True                           False               True   \n",
       "4           False                            True              False   \n",
       "\n",
       "   InternetService_Fiber optic  InternetService_No  ...  \\\n",
       "0                        False               False  ...   \n",
       "1                        False               False  ...   \n",
       "2                        False               False  ...   \n",
       "3                         True               False  ...   \n",
       "4                        False               False  ...   \n",
       "\n",
       "   TechSupport_No internet service  TechSupport_Yes  \\\n",
       "0                            False            False   \n",
       "1                            False            False   \n",
       "2                            False             True   \n",
       "3                            False            False   \n",
       "4                            False            False   \n",
       "\n",
       "   StreamingTV_No internet service  StreamingTV_Yes  \\\n",
       "0                            False            False   \n",
       "1                            False            False   \n",
       "2                            False            False   \n",
       "3                            False             True   \n",
       "4                            False            False   \n",
       "\n",
       "   StreamingMovies_No internet service  StreamingMovies_Yes  \\\n",
       "0                                False                False   \n",
       "1                                False                False   \n",
       "2                                False                False   \n",
       "3                                False                False   \n",
       "4                                False                False   \n",
       "\n",
       "   PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  \\\n",
       "0                  True                                  False   \n",
       "1                 False                                  False   \n",
       "2                 False                                  False   \n",
       "3                  True                                   True   \n",
       "4                 False                                  False   \n",
       "\n",
       "   PaymentMethod_Electronic check  PaymentMethod_Mailed check  \n",
       "0                            True                       False  \n",
       "1                           False                        True  \n",
       "2                           False                       False  \n",
       "3                           False                       False  \n",
       "4                           False                        True  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_upsampled.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)\n",
    "y = X_upsampled['Churn'] #target (dependent variable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "\n",
    "def Report(name,model, X_train, X_test, y_train, y_test):\n",
    "    print(name)\n",
    "    model.fit(X_train, y_train)\n",
    "    pred = model.predict(X_train)\n",
    "    print(f\"accuracy score on training data : {accuracy_score(y_train,pred)} \" )\n",
    "    print(f\"F-1 score on training data : {f1_score(y_train,pred)} \" )\n",
    "    print(f\"recall score on training data : {recall_score(y_train,pred)} \" )\n",
    "    pred_test = model.predict(X_test)\n",
    "    print(f\"accuracy score on test data : {accuracy_score(y_test,pred_test)} \" )\n",
    "    print(f\"F-1 score on test data : {f1_score(y_test,pred_test)} \" )\n",
    "    print(f\"recall score on test data : {recall_score(y_test,pred_test)} \" )\n",
    "    # Calculate confusion matrix\n",
    "    conf_matrix = confusion_matrix(y_test, pred_test)\n",
    "    \n",
    "    # Create a figure and axis\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    plt.imshow(conf_matrix, cmap=plt.cm.Blues)\n",
    "    \n",
    "    # Add color bar\n",
    "    plt.colorbar()\n",
    "    \n",
    "    # Set labels and title\n",
    "    plt.title('Confusion Matrix')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('Actual')\n",
    "    \n",
    "    # Create tick marks\n",
    "    tick_marks = np.arange(len(conf_matrix))\n",
    "    plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])\n",
    "    plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])\n",
    "    \n",
    "    # Add text annotations\n",
    "    for i in range(len(conf_matrix)):\n",
    "        for j in range(len(conf_matrix)):\n",
    "            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')\n",
    "    \n",
    "    # Show the plot\n",
    "    plt.show()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "#parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}\n",
    "def optimization(model, paramneters, n_jobs=-1, cv=5):\n",
    "    clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)\n",
    "    return clf\n",
    "# cv = 5 means having a 5-fold cross validation.\n",
    "# So dataset is divided into 5 subset.\n",
    "# At each iteration, 4 subsets are used in training and the other subset is used as test set.\n",
    "# When 5 iteration completed, the model used all samples as both training and test samples.\n",
    "# n_jobs parameter is used to select how many processors to use. -1 means using all processors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.linear_model import RidgeClassifier\n",
    "from sklearn.linear_model import SGDClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "input = {'X_train' :X_train, \"X_test\":X_test,'y_train': y_train,'y_test' : y_test}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GaussianNB\n",
      "accuracy score on training data : 0.6703309978255617 \n",
      "F-1 score on training data : 0.7366592685515777 \n",
      "recall score on training data : 0.9202025072324012 \n",
      "accuracy score on test data : 0.6613526570048309 \n",
      "F-1 score on test data : 0.7269185820023374 \n",
      "recall score on test data : 0.9093567251461988 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABZhElEQVR4nO3dfXzN9f/H8ec5m53NtnNm2GZhhFyUCMW6Qpar1ZesbynVSClRIVLfb6EJtV9K5CLflotKFyq+kWhR85W5SE2ShNQUm8tthl1/fn9opz656HNOm2153N0+t6/zeb8/78/7c/oee+31vjg2wzAMAQAAAB6yV3QHAAAAUDURSAIAAMArBJIAAADwCoEkAAAAvEIgCQAAAK8QSAIAAMArBJIAAADwCoEkAAAAvEIgCQAAAK8QSAKoFHbs2KGuXbvK5XLJZrNp8eLFZdr+jz/+KJvNprlz55Zpu1VZp06d1KlTp4ruBoAqjEASgNuuXbt033336cILL5S/v7+cTqeuuuoqvfjiizpx4kS53js+Pl5btmzRhAkT9Nprr6ldu3bler9zqX///rLZbHI6nad9H3fs2CGbzSabzabnnnvO4/b37t2rcePGKS0trQx6CwDW+VZ0BwBUDh9++KH++c9/yuFw6K677tIll1yigoICrVmzRqNGjdLWrVs1e/bscrn3iRMnlJqaqn//+98aOnRoudwjKipKJ06cULVq1cql/T/j6+ur48ePa8mSJbrllltMZW+88Yb8/f2Vl5fnVdt79+7VU089pQYNGqh169aWr/v444+9uh8AlCKQBKDdu3erb9++ioqK0qpVq1SnTh132ZAhQ7Rz5059+OGH5Xb/AwcOSJJCQkLK7R42m03+/v7l1v6fcTgcuuqqq/Tmm2+eEkguWLBAsbGxeu+9985JX44fP67q1avLz8/vnNwPwN8XQ9sAlJiYqNzcXCUlJZmCyFKNGzfWww8/7H5dVFSk8ePHq1GjRnI4HGrQoIH+9a9/KT8/33RdgwYNdMMNN2jNmjW64oor5O/vrwsvvFDz58931xk3bpyioqIkSaNGjZLNZlODBg0knRwSLv37740bN042m810Ljk5WVdffbVCQkIUFBSkpk2b6l//+pe7/ExzJFetWqVrrrlGgYGBCgkJUa9evbRt27bT3m/nzp3q37+/QkJC5HK5NGDAAB0/fvzMb+wf3H777froo4+UlZXlPrdx40bt2LFDt99++yn1Dx8+rJEjR6ply5YKCgqS0+lUjx49tHnzZnedzz77TJdffrkkacCAAe4h8tLn7NSpky655BJt2rRJ1157rapXr+5+X/44RzI+Pl7+/v6nPH+3bt1Uo0YN7d271/KzAjg/EEgC0JIlS3ThhRfqyiuvtFT/nnvu0ZgxY9SmTRu98MIL6tixoyZNmqS+ffueUnfnzp26+eabdf3112vy5MmqUaOG+vfvr61bt0qS+vTpoxdeeEGSdNttt+m1117TlClTPOr/1q1bdcMNNyg/P18JCQmaPHmy/vGPf+jzzz8/63WffPKJunXrpv3792vcuHEaMWKE1q5dq6uuuko//vjjKfVvueUWHT16VJMmTdItt9yiuXPn6qmnnrLczz59+shms+n99993n1uwYIGaNWumNm3anFL/hx9+0OLFi3XDDTfo+eef16hRo7RlyxZ17NjRHdQ1b95cCQkJkqRBgwbptdde02uvvaZrr73W3c6hQ4fUo0cPtW7dWlOmTFHnzp1P278XX3xRtWvXVnx8vIqLiyVJL7/8sj7++GNNmzZNkZGRlp8VwHnCAHBey87ONiQZvXr1slQ/LS3NkGTcc889pvMjR440JBmrVq1yn4uKijIkGatXr3af279/v+FwOIxHHnnEfW737t2GJOP//u//TG3Gx8cbUVFRp/Rh7Nixxu//+XrhhRcMScaBAwfO2O/Se8yZM8d9rnXr1kZYWJhx6NAh97nNmzcbdrvduOuuu0653913321q86abbjJq1qx5xnv+/jkCAwMNwzCMm2++2ejSpYthGIZRXFxsREREGE899dRp34O8vDyjuLj4lOdwOBxGQkKC+9zGjRtPebZSHTt2NCQZs2bNOm1Zx44dTedWrFhhSDKefvpp44cffjCCgoKM3r17/+kzAjg/kZEEznM5OTmSpODgYEv1ly1bJkkaMWKE6fwjjzwiSafMpWzRooWuueYa9+vatWuradOm+uGHH7zu8x+Vzq3873//q5KSEkvX7Nu3T2lpaerfv79CQ0Pd5y+99FJdf/317uf8vfvvv9/0+pprrtGhQ4fc76EVt99+uz777DNlZGRo1apVysjIOO2wtnRyXqXdfvKf6eLiYh06dMg9bP/ll19avqfD4dCAAQMs1e3atavuu+8+JSQkqE+fPvL399fLL79s+V4Azi8EksB5zul0SpKOHj1qqf5PP/0ku92uxo0bm85HREQoJCREP/30k+l8/fr1T2mjRo0aOnLkiJc9PtWtt96qq666Svfcc4/Cw8PVt29fvfPOO2cNKkv72bRp01PKmjdvroMHD+rYsWOm8398lho1akiSR8/Ss2dPBQcH6+2339Ybb7yhyy+//JT3slRJSYleeOEFNWnSRA6HQ7Vq1VLt2rX19ddfKzs72/I9L7jgAo8W1jz33HMKDQ1VWlqapk6dqrCwMMvXAji/EEgC5zmn06nIyEh98803Hl33x8UuZ+Lj43Pa84ZheH2P0vl7pQICArR69Wp98sknuvPOO/X111/r1ltv1fXXX39K3b/irzxLKYfDoT59+mjevHlatGjRGbORkjRx4kSNGDFC1157rV5//XWtWLFCycnJuvjiiy1nXqWT748nvvrqK+3fv1+StGXLFo+uBXB+IZAEoBtuuEG7du1Samrqn9aNiopSSUmJduzYYTqfmZmprKws9wrsslCjRg3TCudSf8x6SpLdbleXLl30/PPP69tvv9WECRO0atUqffrpp6dtu7Sf27dvP6Xsu+++U61atRQYGPjXHuAMbr/9dn311Vc6evToaRcolXr33XfVuXNnJSUlqW/fvuratatiYmJOeU+sBvVWHDt2TAMGDFCLFi00aNAgJSYmauPGjWXWPoC/FwJJAHr00UcVGBioe+65R5mZmaeU79q1Sy+++KKkk0Ozkk5ZWf38889LkmJjY8usX40aNVJ2dra+/vpr97l9+/Zp0aJFpnqHDx8+5drSjbn/uCVRqTp16qh169aaN2+eKTD75ptv9PHHH7ufszx07txZ48eP10svvaSIiIgz1vPx8Tkl27lw4UL98ssvpnOlAe/pgm5PjR49Wunp6Zo3b56ef/55NWjQQPHx8Wd8HwGc39iQHIAaNWqkBQsW6NZbb1Xz5s1N32yzdu1aLVy4UP3795cktWrVSvHx8Zo9e7aysrLUsWNHbdiwQfPmzVPv3r3PuLWMN/r27avRo0frpptu0kMPPaTjx49r5syZuuiii0yLTRISErR69WrFxsYqKipK+/fv14wZM1S3bl1dffXVZ2z///7v/9SjRw9FR0dr4MCBOnHihKZNmyaXy6Vx48aV2XP8kd1u1xNPPPGn9W644QYlJCRowIABuvLK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RAMHDlRubq5H/SCQBAAAsMhuK5/DE0eOHNFVV12latWq6aOPPtK3336ryZMnq0aNGu46iYmJmjp1qmbNmqX169crMDBQ3bp1U15enrtOv379tHXrViUnJ2vp0qVavXq1Bg0a5FFfGNoGAACoQp599lnVq1dPc+bMcZ9r2LCh+++GYWjKlCl64okn1KtXL0nS/PnzFR4ersWLF6tv377atm2bli9fro0bN6pdu3aSpGnTpqlnz5567rnnFBkZaakvZCQBAACssv02T7KsjtIdyXNyckxHfn7+abvwwQcfqF27dvrnP/+psLAwXXbZZfrPf/7jLt+9e7cyMjIUExPjPudyudS+fXulpqZKklJTUxUSEuIOIiUpJiZGdrtd69evt/x2EEgCAABUAvXq1ZPL5XIfkyZNOm29H374QTNnzlSTJk20YsUKDR48WA899JDmzZsnScrIyJAkhYeHm64LDw93l2VkZCgsLMxU7uvrq9DQUHcdKxjaBgAAsKg8t//Zs2ePnE6n+7zD4Tht/ZKSErVr104TJ06UJF122WX65ptvNGvWLMXHx5dt5/4EGUkAAIBKwOl0mo4zBZJ16tRRixYtTOeaN2+u9PR0SVJERIQkKTMz01QnMzPTXRYREaH9+/ebyouKinT48GF3HSsIJAEAACyyldMfT1x11VXavn276dz333+vqKgoSScX3kRERGjlypXu8pycHK1fv17R0dGSpOjoaGVlZWnTpk3uOqtWrVJJSYnat29vuS8MbQMAAFjkzXY9Vtr0xPDhw3XllVdq4sSJuuWWW7RhwwbNnj1bs2fPlnRyMdCwYcP09NNPq0mTJmrYsKGefPJJRUZGqnfv3pJOZjC7d++ue++9V7NmzVJhYaGGDh2qvn37Wl6xLRFIAgAAVCmXX365Fi1apMcff1wJCQlq2LChpkyZon79+rnrPProozp27JgGDRqkrKwsXX311Vq+fLn8/f3ddd544w0NHTpUXbp0kd1uV1xcnKZOnepRXwgkAQAALKosX5F4ww036IYbbjhrmwkJCUpISDhjndDQUC1YsMDje/8ecyQBAADgFTKSAAAAFpXn9j9VERlJAAAAeIWMJAAAgEV2m032Mk4hlnV75xIZSQAAAHiFjCQAAIBFzJE0I5AEAACwqLJs/1NZMLQNAAAAr5CRBAAAsIihbTMykgAAAPAKGUkAAACL2P7HjIwkAAAAvEJGEgAAwCLbr0dZt1lVkZEEAACAV8hIAgAAWMQ+kmYEkgAAABbZbSePsm6zqmJoGwAAAF4hIwkAAGARQ9tmZCQBAADgFTKSAAAAHqjCCcQyR0YSAAAAXiEjCQAAYBFzJM0sBZIffPCB5Qb/8Y9/eN0ZAAAAVB2WAsnevXtbasxms6m4uPiv9AcAAKDSYh9JM0uBZElJSXn3AwAAoNJjaNuMxTYAAADwileLbY4dO6aUlBSlp6eroKDAVPbQQw+VSccAAAAqG9uvR1m3WVV5HEh+9dVX6tmzp44fP65jx44pNDRUBw8eVPXq1RUWFkYgCQAAcJ7weGh7+PDhuvHGG3XkyBEFBARo3bp1+umnn9S2bVs999xz5dFHAACASsFus5XLUVV5HEimpaXpkUcekd1ul4+Pj/Lz81WvXj0lJibqX//6V3n0EQAAAJWQx4FktWrVZLefvCwsLEzp6emSJJfLpT179pRt7wAAACoRm618jqrK4zmSl112mTZu3KgmTZqoY8eOGjNmjA4ePKjXXntNl1xySXn0EQAAAJWQxxnJiRMnqk6dOpKkCRMmqEaNGho8eLAOHDig2bNnl3kHAQAAKovSfSTL+qiqPM5ItmvXzv33sLAwLV++vEw7BAAAgKrBq30kAQAAzkflMaexCickPQ8kGzZseNYU7A8//PCXOgQAAFBZlcd2PVV5+x+PA8lhw4aZXhcWFuqrr77S8uXLNWrUqLLqFwAAACo5jwPJhx9++LTnp0+fri+++OIvdwgAAKCyYmjbzONV22fSo0cPvffee2XVHAAAACq5Mlts8+677yo0NLSsmgMAAKh0ymO7nvNq+5/LLrvM9MCGYSgjI0MHDhzQjBkzyrRz54sBVzSQ0+ms6G4A+J3HHppc0V0A8AdGcUFFdwF/4HEg2atXL1MgabfbVbt2bXXq1EnNmjUr084BAABUJnaV4bzA37VZVXkcSI4bN64cugEAAICqxuMg2MfHR/v37z/l/KFDh+Tj41MmnQIAAKiM+IpEM48DScMwTns+Pz9ffn5+f7lDAAAAlZXNJtnL+PA0jhw3btwpgejvpxfm5eVpyJAhqlmzpoKCghQXF6fMzExTG+np6YqNjVX16tUVFhamUaNGqaioyOP3w/LQ9tSpUyWdjMRfeeUVBQUFucuKi4u1evVq5kgCAACcAxdffLE++eQT92tf399CuuHDh+vDDz/UwoUL5XK5NHToUPXp00eff/65pJNxW2xsrCIiIrR27Vrt27dPd911l6pVq6aJEyd61A/LgeQLL7wg6WRGctasWaZhbD8/PzVo0ECzZs3y6OYAAABVSWkWsazb9JSvr68iIiJOOZ+dna2kpCQtWLBA1113nSRpzpw5at68udatW6cOHTro448/1rfffqtPPvlE4eHhat26tcaPH6/Ro0dr3LhxHo0wWw4kd+/eLUnq3Lmz3n//fdWoUcPyTQAAAHB2OTk5ptcOh0MOh+O0dXfs2KHIyEj5+/srOjpakyZNUv369bVp0yYVFhYqJibGXbdZs2aqX7++UlNT1aFDB6Wmpqply5YKDw931+nWrZsGDx6srVu36rLLLrPcZ4/nSH766acEkQAA4LxUnott6tWrJ5fL5T4mTZp02j60b99ec+fO1fLlyzVz5kzt3r1b11xzjY4ePaqMjAz5+fkpJCTEdE14eLgyMjIkSRkZGaYgsrS8tMwTHm//ExcXpyuuuEKjR482nU9MTNTGjRu1cOFCT5sEAAA47+3Zs8f0BSVnykb26NHD/fdLL71U7du3V1RUlN555x0FBASUez9/z+OM5OrVq9WzZ89Tzvfo0UOrV68uk04BAABURmW9Yvv3cy6dTqfpOFMg+UchISG66KKLtHPnTkVERKigoEBZWVmmOpmZme45lREREaes4i59fbp5l2d9PzyqLSk3N/e0kzCrVat2ytg+AAAAyldubq527dqlOnXqqG3btqpWrZpWrlzpLt++fbvS09MVHR0tSYqOjtaWLVtM+4InJyfL6XSqRYsWHt3b40CyZcuWevvtt085/9Zbb3l8cwAAgKrEZiufwxMjR45USkqKfvzxR61du1Y33XSTfHx8dNttt8nlcmngwIEaMWKEPv30U23atEkDBgxQdHS0OnToIEnq2rWrWrRooTvvvFObN2/WihUr9MQTT2jIkCGWs6ClPJ4j+eSTT6pPnz7atWuXe1n5ypUrtWDBAr377rueNgcAAFBl2G022cv4m2g8be/nn3/WbbfdpkOHDql27dq6+uqrtW7dOtWuXVvSyS0b7Xa74uLilJ+fr27dumnGjBnu6318fLR06VINHjxY0dHRCgwMVHx8vBISEjzuu8eB5I033qjFixdr4sSJevfddxUQEKBWrVpp1apVCg0N9bgDAAAAsO6tt946a7m/v7+mT5+u6dOnn7FOVFSUli1b9pf74nEgKUmxsbGKjY2VdHLPozfffFMjR47Upk2bVFxc/Jc7BQAAUBnZ5cW8QAttVlVe93316tWKj49XZGSkJk+erOuuu07r1q0ry74BAACgEvMoI5mRkaG5c+cqKSlJOTk5uuWWW5Sfn6/Fixez0AYAAPztebM4xkqbVZXljOSNN96opk2b6uuvv9aUKVO0d+9eTZs2rTz7BgAAgErMckbyo48+0kMPPaTBgwerSZMm5dknAACASsmucli1raqbkrSckVyzZo2OHj2qtm3bqn379nrppZd08ODB8uwbAAAAKjHLgWSHDh30n//8R/v27dN9992nt956S5GRkSopKVFycrKOHj1anv0EAACocJVhQ/LKxONV24GBgbr77ru1Zs0abdmyRY888oieeeYZhYWF6R//+Ed59BEAAKBSKM/v2q6K/tLWRU2bNlViYqJ+/vlnvfnmm2XVJwAAAFQBXm1I/kc+Pj7q3bu3evfuXRbNAQAAVEo2m+dfaWilzaqqKm+mDgAAgApUJhlJAACA8wEbkpuRkQQAAIBXyEgCAABYVB6rrM/bVdsAAAA4f5GRBAAAsMj265+ybrOqIpAEAACwiKFtM4a2AQAA4BUykgAAABaRkTQjIwkAAACvkJEEAACwyGazyVbmX5FYdVOSZCQBAADgFTKSAAAAFjFH0oyMJAAAALxCRhIAAMAim+3kUdZtVlUEkgAAABbZbTbZyzjyK+v2ziWGtgEAAOAVMpIAAAAWsdjGjIwkAAAAvEJGEgAAwKpyWGwjMpIAAAA435CRBAAAsMgum+xlnEIs6/bOJTKSAAAA8AoZSQAAAIvYkNyMQBIAAMAitv8xY2gbAAAAXiEjCQAAYBFfkWhGRhIAAABeISMJAABgEYttzMhIAgAAwCtkJAEAACyyqxzmSLIhOQAAAM43ZCQBAAAsYo6kGYEkAACARXaV/XBuVR4ersp9BwAAOO8988wzstlsGjZsmPtcXl6ehgwZopo1ayooKEhxcXHKzMw0XZeenq7Y2FhVr15dYWFhGjVqlIqKijy6N4EkAACARTabrVwOb23cuFEvv/yyLr30UtP54cOHa8mSJVq4cKFSUlK0d+9e9enTx11eXFys2NhYFRQUaO3atZo3b57mzp2rMWPGeHR/AkkAAIAqKDc3V/369dN//vMf1ahRw30+OztbSUlJev7553Xdddepbdu2mjNnjtauXat169ZJkj7++GN9++23ev3119W6dWv16NFD48eP1/Tp01VQUGC5DwSSAAAAFtnK6ZCknJwc05Gfn3/WvgwZMkSxsbGKiYkxnd+0aZMKCwtN55s1a6b69esrNTVVkpSamqqWLVsqPDzcXadbt27KycnR1q1bLb8fBJIAAACVQL169eRyudzHpEmTzlj3rbfe0pdffnnaOhkZGfLz81NISIjpfHh4uDIyMtx1fh9ElpaXllnFqm0AAACL7LZy2JD81/b27Nkjp9PpPu9wOE5bf8+ePXr44YeVnJwsf3//Mu2Lp8hIAgAAVAJOp9N0nCmQ3LRpk/bv3682bdrI19dXvr6+SklJ0dSpU+Xr66vw8HAVFBQoKyvLdF1mZqYiIiIkSREREaes4i59XVrHCgJJAAAAD5TH/EhPdOnSRVu2bFFaWpr7aNeunfr16+f+e7Vq1bRy5Ur3Ndu3b1d6erqio6MlSdHR0dqyZYv279/vrpOcnCyn06kWLVpY7gtD2wAAABZVhm+2CQ4O1iWXXGI6FxgYqJo1a7rPDxw4UCNGjFBoaKicTqcefPBBRUdHq0OHDpKkrl27qkWLFrrzzjuVmJiojIwMPfHEExoyZMgZM6GnQyAJAADwN/PCCy/IbrcrLi5O+fn56tatm2bMmOEu9/Hx0dKlSzV48GBFR0crMDBQ8fHxSkhI8Og+BJIAAAAW/dUNxM/U5l/12WefmV77+/tr+vTpmj59+hmviYqK0rJly/7SfZkjCQAAAK+QkQQAALDIrrLPwlXlrF5V7jsAAAAqEBlJAAAAiyrrHMmKQkYSAAAAXiEjCQAAYJG3m4j/WZtVFRlJAAAAeIWMJAAAgEXMkTQjkAQAALCI7X/MqnLfAQAAUIHISAIAAFjE0LYZGUkAAAB4hYwkAACARWz/Y0ZGEgAAAF4hIwkAAGCRzXbyKOs2qyoykgAAAPAKGUkAAACL7LLJXsazGsu6vXOJQBIAAMAihrbNGNoGAACAV8hIAgAAWGT79U9Zt1lVkZEEAACAV8hIAgAAWMQcSTMykgAAAPAKGUkAAACLbOWw/Q9zJAEAAHDeISMJAABgEXMkzQgkAQAALCKQNGNoGwAAAF4hIwkAAGARG5KbkZEEAACAV8hIAgAAWGS3nTzKus2qiowkAAAAvEJGEgAAwCLmSJqRkQQAAIBXyEgCAABYxD6SZgSSAAAAFtlU9kPRVTiOZGgbAAAA3iEjCQAAYBHb/5iRkQQAAIBXyEgCAABYxPY/ZmQkAQAA4BUykmdhs9m0aNEi9e7du6K7gkrG1y75/PoLpCGpsPjk/5aW2W2/rcIrMaSikt/KAfx1QdUdGvvADfrHda1Uu0aQNm//WSMT39Wmb9MlSf++r6f+2a2N6kbUUEFhsb7alq5xLy3Rxm9+crexcMp9anXRBaodGqwjOcf16frtemLqf7XvQHZFPRaqALb/MasUGcnU1FT5+PgoNjbW42sbNGigKVOmlH2nLFi9erVuvPFGRUZGymazafHixRXSD5xb1X4NFAuKTx4lhuTn81t5aeBYWm7IXA7gr5s55nZd16GZ7n5intrdMlGfpH6nD2c9qMjaLknSzp/2a/izC9XunxPVZcDz+mnvYS2ZMVS1agS521i98XvdMfpVtbopQbePekUX1qulBf83sKIeCaiSKkUgmZSUpAcffFCrV6/W3r17K7o7lh07dkytWrXS9OnTK7orOIfstt8yjIZ++7vvr5+mEuPk8ftyWzms8gPOV/6OaurdpbX+PWWxPv9yl37Yc1ATXl6mXXsO6N5/XiNJenv5F/p0/Xb9+MshbfshQ6Mnvy9XcIAuaRLpbmfaG59qw5Yflb7viNZt3q3n5iTripYN5OtbKX40opKyldNRVVX4pyU3N1dvv/22Bg8erNjYWM2dO/eUOkuWLNHll18uf39/1apVSzfddJMkqVOnTvrpp580fPhw2Ww22X7NDY8bN06tW7c2tTFlyhQ1aNDA/Xrjxo26/vrrVatWLblcLnXs2FFffvmlR33v0aOHnn76aXd/cH440xDEmQJFH5tk/BpcAvjrfH3s8vX1UV5Boel8Xn6hrrys0Sn1q/n6aGCfq5R19Li2fP/Ladus4ayuvj3aad3m3SoqKimXfuPvwS6b7LYyPjwMJWfOnKlLL71UTqdTTqdT0dHR+uijj9zleXl5GjJkiGrWrKmgoCDFxcUpMzPT1EZ6erpiY2NVvXp1hYWFadSoUSoqKvLi/ahg77zzjpo1a6amTZvqjjvu0KuvvirD+O0n7ocffqibbrpJPXv21FdffaWVK1fqiiuukCS9//77qlu3rhISErRv3z7t27fP8n2PHj2q+Ph4rVmzRuvWrVOTJk3Us2dPHT16tMyfsVR+fr5ycnJMB6qeEuO37KNkng/5+3MOn5OHr/3kEDeAspF7PF/rNv+gx+/toTq1XbLbberb83K1v7ShImo53fV6XHOJDnw+WVnrX9CDd3TWDfe/pENZx0xtPf1QLx1cO1l7UxJVr06o/jl89rl+HMBjdevW1TPPPKNNmzbpiy++0HXXXadevXpp69atkqThw4dryZIlWrhwoVJSUrR371716dPHfX1xcbFiY2NVUFCgtWvXat68eZo7d67GjBnjcV8qPJBMSkrSHXfcIUnq3r27srOzlZKS4i6fMGGC+vbtq6eeekrNmzdXq1at9Pjjj0uSQkND5ePjo+DgYEVERCgiIsLyfa+77jrdcccdatasmZo3b67Zs2fr+PHjpnuXtUmTJsnlcrmPevXqldu9UH4Kfw0K/X1/CxT/mG0sMX6bI1lsSNWYIwmUqbufmC+bTfrh4wnKXj9FQ27rqHeWf6GS330YUzZ+r/Z9J6lz/+f18dpv9Xri3ar9uzmSkvTC/E/Uoe+zir3/JRUXl+iV8Xee60dBFVMZhrZvvPFG9ezZU02aNNFFF12kCRMmKCgoSOvWrVN2draSkpL0/PPP67rrrlPbtm01Z84crV27VuvWrZMkffzxx/r222/1+uuvq3Xr1urRo4fGjx+v6dOnq6CgwKO+VGgguX37dm3YsEG33XabJMnX11e33nqrkpKS3HXS0tLUpUuXMr93Zmam7r33XjVp0kQul0tOp1O5ublKT08v83uVevzxx5Wdne0+9uzZU273QvkxdDJAzCuS8ot/yzYaxqn1SudISr+t8gbw1+3++aC63vOiakaPUJMeT+qaO59TNV8f7f7loLvO8bwC/bDnoDZs+VGDn1qgouISxd90pamdQ1nHtDN9v1at/053PTZHPa65RO0vbXiuHweQpFNGLfPz8//0muLiYr311ls6duyYoqOjtWnTJhUWFiomJsZdp1mzZqpfv75SU1MlnVzk3LJlS4WHh7vrdOvWTTk5Oe6splUVuv1PUlKSioqKFBn52+RnwzDkcDj00ksvyeVyKSAgwON27Xa7aXhckgoLzXNp4uPjdejQIb344ouKioqSw+FQdHS0x5G4JxwOhxwOR7m1j4pTugDnbGw2sQcQUMaO5xXoeF6BQoIDFHNlc/17yn/PWNdus8lR7cw/9uy/TnT2O0sdoFxWx/za3h9HKseOHatx48ad9pItW7YoOjpaeXl5CgoK0qJFi9SiRQulpaXJz89PISEhpvrh4eHKyMiQJGVkZJiCyNLy0jJPVNinpaioSPPnz9fkyZPVtWtXU1nv3r315ptv6v7779ell16qlStXasCAAadtx8/PT8XF5glotWvXVkZGhgzDcC/ASUtLM9X5/PPPNWPGDPXs2VOStGfPHh08eFDAnyldVGMYJ4NDX/vJ+LD41yDR1y4V/7qS2ybJx37yf4uZvw+UmZjo5rLZpO9/3K9G9Wpr4vDe+n53puZ/kKrq/n4afU83fZiyRRkHs1UzJEj33XKtIsNC9H7yyUWVl18SpbYXR2ntV7uUdfS4GtatrbEPxGpX+gGt/3p3BT8dzld79uyR0/nbPN+zJZ+aNm2qtLQ0ZWdn691331V8fHy5Ts87kwoLJJcuXaojR45o4MCBcrlcprK4uDglJSXp/vvv19ixY9WlSxc1atRIffv2VVFRkZYtW6bRo0dLOrmP5OrVq9W3b185HA7VqlVLnTp10oEDB5SYmKibb75Zy5cv10cffWT6j9OkSRO99tprateunXJycjRq1CiPs5+5ubnauXOn+/Xu3buVlpam0NBQ1a9f/y+8O6jsfO2//UJabJizkTaduq9kQTHJSKAsuYL8lfDgP3RBeIgOZx/Xf1emaez0JSoqKpGPvURNG4Trjhvbq2ZIoA5nH9cXW39SzN0vaNsPJ7Mtx/MK1eu6Vnri/lgFBvgp42C2Pl67Tc/+51UVFHq+chXnj/L8isTSVdhW+Pn5qXHjxpKktm3bauPGjXrxxRd16623qqCgQFlZWaasZGZmpnstSUREhDZs2GBqr3RVtyfrTaQKDCSTkpIUExNzShApnQwkExMT9fXXX6tTp05auHChxo8fr2eeeUZOp1PXXnutu25CQoLuu+8+NWrUSPn5+TIMQ82bN9eMGTM0ceJEjR8/XnFxcRo5cqRmz55tuv+gQYPUpk0b1atXTxMnTtTIkSM9eoYvvvhCnTt3dr8eMWKEpJPD5qfbxgh/D6WB4ZkUknkEyt17yV/pveSvTluWX1CkviNfOev1W3fuVY/7ppVH14AKUVJSovz8fLVt21bVqlXTypUrFRcXJ+nkmpT09HRFR0dLkqKjozVhwgTt379fYWFhkqTk5GQ5nU61aNHCo/vajD9OJsQ5k5OTI5fLpcxD2ZZ/AwFwbtS4fGhFdwHAHxjFBcrf8h9lZ5/7n5ulP7NXpqUrKLhs7517NEddWte3/FyPP/64evToofr16+vo0aNasGCBnn32Wa1YsULXX3+9Bg8erGXLlmnu3LlyOp168MEHJUlr166VdHKBTuvWrRUZGanExERlZGTozjvv1D333KOJEyd61HdmFAMAAFhUjmttLNu/f7/uuusu7du3Ty6XS5deeqk7iJSkF154QXa7XXFxccrPz1e3bt00Y8YM9/U+Pj5aunSpBg8erOjoaAUGBio+Pl4JCQme952MZMUhIwlUXmQkgcqnMmQkV5VTRvI6DzKSlQkZSQAAAKsqQ0qyEqnwb7YBAABA1URGEgAAwKLy3P6nKiIjCQAAAK+QkQQAALDIZvv1K2/LuM2qiowkAAAAvEJGEgAAwCIWbZsRSAIAAFhFJGnC0DYAAAC8QkYSAADAIrb/MSMjCQAAAK+QkQQAALCI7X/MyEgCAADAK2QkAQAALGLRthkZSQAAAHiFjCQAAIBVpCRNCCQBAAAsYvsfM4a2AQAA4BUykgAAABax/Y8ZGUkAAAB4hYwkAACARay1MSMjCQAAAK+QkQQAALCKlKQJGUkAAAB4hYwkAACARewjaUZGEgAAAF4hIwkAAGAR+0iaEUgCAABYxFobM4a2AQAA4BUykgAAAFaRkjQhIwkAAACvkJEEAACwiO1/zMhIAgAAwCtkJAEAACxi+x8zMpIAAADwChlJAAAAi1i0bUYgCQAAYBWRpAlD2wAAAPAKGUkAAACL2P7HjIwkAAAAvEJGEgAAwKpy2P6nCickyUgCAADAO2QkAQAALGLRthkZSQAAAHiFQBIAAMAqWzkdHpg0aZIuv/xyBQcHKywsTL1799b27dtNdfLy8jRkyBDVrFlTQUFBiouLU2ZmpqlOenq6YmNjVb16dYWFhWnUqFEqKiryqC8EkgAAABbZyumPJ1JSUjRkyBCtW7dOycnJKiwsVNeuXXXs2DF3neHDh2vJkiVauHChUlJStHfvXvXp08ddXlxcrNjYWBUUFGjt2rWaN2+e5s6dqzFjxnj2fhiGYXh0BcpMTk6OXC6XMg9ly+l0VnR3APxOjcuHVnQXAPyBUVyg/C3/UXb2uf+5WfozO21XpoKDy/beR4/mqHWjcK+f68CBAwoLC1NKSoquvfZaZWdnq3bt2lqwYIFuvvlmSdJ3332n5s2bKzU1VR06dNBHH32kG264QXv37lV4eLgkadasWRo9erQOHDggPz8/S/cmIwkAAGCRzVY+h3QyWP39kZ+fb6lP2dnZkqTQ0FBJ0qZNm1RYWKiYmBh3nWbNmql+/fpKTU2VJKWmpqply5buIFKSunXrppycHG3dutXy+0EgCQAAUAnUq1dPLpfLfUyaNOlPrykpKdGwYcN01VVX6ZJLLpEkZWRkyM/PTyEhIaa64eHhysjIcNf5fRBZWl5aZhXb/wAAAFhUntv/7NmzxzS07XA4/vTaIUOG6JtvvtGaNWvKuFfWkJEEAACoBJxOp+n4s0By6NChWrp0qT799FPVrVvXfT4iIkIFBQXKysoy1c/MzFRERIS7zh9XcZe+Lq1jBYEkAACAVZVg+x/DMDR06FAtWrRIq1atUsOGDU3lbdu2VbVq1bRy5Ur3ue3btys9PV3R0dGSpOjoaG3ZskX79+9310lOTpbT6VSLFi0s94WhbQAAgCpkyJAhWrBggf773/8qODjYPafR5XIpICBALpdLAwcO1IgRIxQaGiqn06kHH3xQ0dHR6tChgySpa9euatGihe68804lJiYqIyNDTzzxhIYMGWJpSL0UgSQAAIBF3uz7aKVNT8ycOVOS1KlTJ9P5OXPmqH///pKkF154QXa7XXFxccrPz1e3bt00Y8YMd10fHx8tXbpUgwcPVnR0tAIDAxUfH6+EhASP+kIgCQAAYJFNv23XU5ZtesLKFuD+/v6aPn26pk+ffsY6UVFRWrZsmYd3N2OOJAAAALxCRhIAAMCi8tz+pyoiIwkAAACvkJEEAACw6PdfaViWbVZVZCQBAADgFTKSAAAAljFL8vfISAIAAMArZCQBAAAsYo6kGYEkAACARQxsmzG0DQAAAK+QkQQAALCIoW0zMpIAAADwChlJAAAAi2y//inrNqsqMpIAAADwChlJAAAAq1i2bUJGEgAAAF4hIwkAAGARCUkzAkkAAACL2P7HjKFtAAAAeIWMJAAAgEVs/2NGRhIAAABeISMJAABgFattTMhIAgAAwCtkJAEAACwiIWlGRhIAAABeISMJAABgEftImhFIAgAAWFb22/9U5cFthrYBAADgFTKSAAAAFjG0bUZGEgAAAF4hkAQAAIBXCCQBAADgFeZIAgAAWMQcSTMykgAAAPAKGUkAAACLbOWwj2TZ70t57hBIAgAAWMTQthlD2wAAAPAKGUkAAACLbCr7LzSswglJMpIAAADwDhlJAAAAq0hJmpCRBAAAgFfISAIAAFjE9j9mZCQBAADgFTKSAAAAFrGPpBkZSQAAgCpm9erVuvHGGxUZGSmbzabFixebyg3D0JgxY1SnTh0FBAQoJiZGO3bsMNU5fPiw+vXrJ6fTqZCQEA0cOFC5ubke9YNAEgAAwCJbOR2eOnbsmFq1aqXp06eftjwxMVFTp07VrFmztH79egUGBqpbt27Ky8tz1+nXr5+2bt2q5ORkLV26VKtXr9agQYM86gdD2wAAAFZVku1/evTooR49epy2zDAMTZkyRU888YR69eolSZo/f77Cw8O1ePFi9e3bV9u2bdPy5cu1ceNGtWvXTpI0bdo09ezZU88995wiIyMt9YOMJAAAQCWQk5NjOvLz871qZ/fu3crIyFBMTIz7nMvlUvv27ZWamipJSk1NVUhIiDuIlKSYmBjZ7XatX7/e8r0IJAEAACyyldMfSapXr55cLpf7mDRpkld9zMjIkCSFh4ebzoeHh7vLMjIyFBYWZir39fVVaGiou44VDG0DAABUAnv27JHT6XS/djgcFdgba8hIAgAAWFS6/U9ZH5LkdDpNh7eBZEREhCQpMzPTdD4zM9NdFhERof3795vKi4qKdPjwYXcdK8hIViDDMCRJR3NyKrgnAP7IKC6o6C4A+IPSz2Xpz8+KkFMOP7PLus2GDRsqIiJCK1euVOvWrd33WL9+vQYPHixJio6OVlZWljZt2qS2bdtKklatWqWSkhK1b9/e8r0IJCvQ0aNHJUmNG9ar4J4AAFB1HD16VC6X65ze08/PTxEREWpSTj+zIyIi5OfnZ7l+bm6udu7c6X69e/dupaWlKTQ0VPXr19ewYcP09NNPq0mTJmrYsKGefPJJRUZGqnfv3pKk5s2bq3v37rr33ns1a9YsFRYWaujQoerbt6/lFduSZDMqMqw/z5WUlGjv3r0KDg6WrSpvaw9JJ3/bq1ev3ilzXABULD6bfx+GYejo0aOKjIyU3X7uZ+fl5eWpoKB8Riv8/Pzk7+9vuf5nn32mzp07n3I+Pj5ec+fOlWEYGjt2rGbPnq2srCxdffXVmjFjhi666CJ33cOHD2vo0KFasmSJ7Ha74uLiNHXqVAUFBVnuB4EkUEZycnLkcrmUnZ3NDyugEuGzCZQfFtsAAADAKwSSAAAA8AqBJFBGHA6Hxo4dWyX2/QLOJ3w2gfLDHEkAAAB4hYwkAAAAvEIgCQAAAK8QSAIAAMArBJLAWfTv39/9LQCS1KlTJw0bNuyc9+Ozzz6TzWZTVlbWOb83UBnx2QQqBwJJVDn9+/eXzWaTzWaTn5+fGjdurISEBBUVFZX7vd9//32NHz/eUt1z/QMmLy9PQ4YMUc2aNRUUFKS4uDhlZmaek3sDEp/NM5k9e7Y6deokp9NJ0Im/HQJJVEndu3fXvn37tGPHDj3yyCMaN26c/u///u+0dcvy66xCQ0MVHBxcZu2VpeHDh2vJkiVauHChUlJStHfvXvXp06eiu4XzDJ/NUx0/flzdu3fXv/71r4ruClDmCCRRJTkcDkVERCgqKkqDBw9WTEyMPvjgA0m/DXlNmDBBkZGRatq0qSRpz549uuWWWxQSEqLQ0FD16tVLP/74o7vN4uJijRgxQiEhIapZs6YeffRR/XF3rD8On+Xn52v06NGqV6+eHA6HGjdurKSkJP3444/u70CtUaOGbDab+vfvL+nkd6xPmjRJDRs2VEBAgFq1aqV3333XdJ9ly5bpoosuUkBAgDp37mzq5+lkZ2crKSlJzz//vK677jq1bdtWc+bM0dq1a7Vu3Tov3mHAO3w2TzVs2DA99thj6tChg4fvJlD5EUjibyEgIMCU3Vi5cqW2b9+u5ORkLV26VIWFherWrZuCg4P1v//9T59//rmCgoLUvXt393WTJ0/W3Llz9eqrr2rNmjU6fPiwFi1adNb73nXXXXrzzTc1depUbdu2TS+//LKCgoJUr149vffee5Kk7du3a9++fXrxxRclSZMmTdL8+fM1a9Ysbd26VcOHD9cdd9yhlJQUSSd/qPbp00c33nij0tLSdM899+ixxx47az82bdqkwsJCxcTEuM81a9ZM9evXV2pqqudvKFBGzvfPJvC3ZwBVTHx8vNGrVy/DMAyjpKTESE5ONhwOhzFy5Eh3eXh4uJGfn+++5rXXXjOaNm1qlJSUuM/l5+cbAQEBxooVKwzDMIw6deoYiYmJ7vLCwkKjbt267nsZhmF07NjRePjhhw3DMIzt27cbkozk5OTT9vPTTz81JBlHjhxxn8vLyzOqV69urF271lR34MCBxm233WYYhmE8/vjjRosWLUzlo0ePPqWt33vjjTcMPz+/U85ffvnlxqOPPnraa4Cyxmfz7E53X6Cq863AGBbw2tKlSxUUFKTCwkKVlJTo9ttv17hx49zlLVu2lJ+fn/v15s2btXPnzlPmUOXl5WnXrl3Kzs7Wvn371L59e3eZr6+v2rVrd8oQWqm0tDT5+PioY8eOlvu9c+dOHT9+XNdff73pfEFBgS677DJJ0rZt20z9kKTo6GjL9wAqEp9N4PxCIIkqqXPnzpo5c6b8/PwUGRkpX1/z/5UDAwNNr3Nzc9W2bVu98cYbp7RVu3Ztr/oQEBDg8TW5ubmSpA8//FAXXHCBqeyvfA9wRESECgoKlJWVpZCQEPf5zMxMRUREeN0u4Ck+m8D5hUASVVJgYKAaN25suX6bNm309ttvKywsTE6n87R16tSpo/Xr1+vaa6+VJBUVFWnTpk1q06bNaeu3bNlSJSUlSklJMc1NLFWadSkuLnafa9GihRwOh9LT08+YLWnevLl7cUKpP1sw07ZtW1WrVk0rV65UXFycpJPzv9LT08mY4JziswmcX1hsg/NCv379VKtWLfXq1Uv/+9//tHv3bn322Wd66KGH9PPPP0uSHn74YT3zzDNavHixvvvuOz3wwANn3e+tQYMGio+P1913363Fixe723znnXckSVFRUbLZbFq6dKkOHDig3NxcBQcHa+TIkRo+fLjmzZunXbt26csvv9S0adM0b948SdL999+vHTt2aNSoUdq+fbsWLFiguXPnnvX5XC6XBg4cqBEjRujTTz/Vpk2bNGDAAEVHR7NSFJXa3/2zKUkZGRlKS0vTzp07JUlbtmxRWlqaDh8+/NfePKAyqOhJmoCnfj+h35Pyffv2GXfddZdRq1Ytw+FwGBdeeKFx7733GtnZ2YZhnJzA//DDDxtOp9MICQkxRowYYdx1111nnNBvGIZx4sQJY/jw4UadOnUMPz8/o3Hjxsarr77qLk9ISDAiIiIMm81mxMfHG4ZxchHClClTjKZNmxrVqlUzateubXTr1s1ISUlxX7dkyRKjcePGhsPhMK655hrj1Vdf/dNJ+idOnDAeeOABo0aNGkb16tWNm266ydi3b99Z30ugLPHZPL2xY8cakk455syZc7a3E6gSbIZxhtnKAAAAwFkwtA0AAACvEEgCAADAKwSSAAAA8AqBJAAAALxCIAkAAACvEEgCAADAKwSSAAAA8AqBJAAAALxCIAngvNe/f3/17t3b/bpTp04aNmzYOe/HZ599JpvNdtav/wOAyoRAEkCl1b9/f9lsNtlsNvn5+alx48ZKSEhQUVFRud73/fff1/jx4y3VJfgDcD7zregOAMDZdO/eXXPmzFF+fr6WLVumIUOGqFq1anr88cdN9QoKCuTn51cm9wwNDS2TdgDg746MJIBKzeFwKCIiQlFRURo8eLBiYmL0wQcfuIejJ0yYoMjISDVt2lSStGfPHt1yyy0KCQlRaGioevXqpR9//NHdXnFxsUaMGKGQkBDVrFlTjz76qAzDMN3zj0Pb+fn5Gj16tOrVqyeHw6HGjRsrKSlJP/74ozp37ixJqlGjhmw2m/r37y9JKikp0aRJk9SwYUMFBASoVatWevfdd033WbZsmS666CIFBASoc+fOpn4CQFVAIAmgSgkICFBBQYEkaeXKldq+fbuSk5O1dOlSFRYWqlu3bgoODtb//vc/ff755woKClL37t3d10yePFlz587Vq6++qjVr1ujw4cNatGjRWe9511136c0339TUqVO1bds2vfzyywoKClK9evX03nvvSZK2b9+uffv26cUXX5QkTZo0SfPnz9esWbO0detWDR8+XHfccYdSUlIknQx4+/TpoxtvvFFpaWm655579Nhjj5XX2wYA5YKhbQBVgmEYWrlypVasWKEHH3xQBw4cUGBgoF555RX3kPbrr7+ukpISvfLKK7LZbJKkOXPmKCQkRJ999pm6du2qKVOm6PHHH1efPn0kSbNmzdKKFSvOeN/vv/9e77zzjpKTkxUTEyNJuvDCC93lpcPgYWFhCgkJkXQygzlx4kR98sknio6Odl+zZs0avfzyy+rYsaNmzpypRo0aafLkyZKkpk2basuWLXr22WfL8F0DgPJFIAmgUlu6dKmCgoJUWFiokpIS3X777Ro3bpyGDBmili1bmuZFbt68WTt37lRwcLCpjby8PO3atUvZ2dnat2+f2rdv7y7z9fVVu3btThneLpWWliYfHx917NjRcp937typ48eP6/rrrzedLygo0GWXXSZJ2rZtm6kfktxBJwBUFQSSACq1zp07a+bMmfLz81NkZKR8fX/7ZyswMNBUNzc3V23bttUbb7xxSju1a9f26v4BAQEeX5ObmytJ+vDDD3XBBReYyhwOh1f9AIDKiEASQKUWGBioxo0bW6rbpk0bvf322woLC5PT6TxtnTp16mj9+vW69tprJUlFRUXatGmT2rRpc9r6LVu2VElJiVJSUtxD279XmhEtLi52n2vRooUcDofS09PPmMls3ry5PvjgA9O5devW/flDAkAlwmIbAH8b/fr1U61atdSrVy/973//0+7du/XZZ5/poYce0s8//yxJevjhh/XMM89o8eLF+u677/TAAw+cdQ/IBg0aKD4+XnfffbcWL17sbvOdd96RJEVFRclms2np0qU6cOCAcnNzFRwcrJEjR2r48OGaN2+edu3apS+//FLTpk3TvHnzJEn333+/duzYoVGjRmn79u1asGCB5s6dW95vEQCUKQJJAH8b1atX1+rVq1W/fn316dNHzZs318CBA5WXl+fOUD7yyCO68847FR8fr+joaAUHB+umm246a7szZ87UzTffrAceeEDNmjXTvffeq2PHjkmSLrjgAj311FN67LHHFB4erqFDh0qSxo8fryeffFKTJk1S8+bN1b17d3344Ydq2LChJKl+/fp67733tHjxYrVq1UqzZs3SxIkTy/HdAYCyZzPONMMcAAAAOAsykgAAAPAKgSQAAAC8QiAJAAAArxBIAgAAwCsEkgAAAPAKgSQAAAC8QiAJAAAArxBIAgAAwCsEkgAAAPAKgSQAAAC8QiAJAAAAr/w/i0dCYgljeFUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "Report(\"GaussianNB\", GaussianNB(), **input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AdaBoostClassifier\n",
      "accuracy score on training data : 0.7672143029717323 \n",
      "F-1 score on training data : 0.7719256716771214 \n",
      "recall score on training data : 0.7861620057859209 \n",
      "accuracy score on test data : 0.7671497584541063 \n",
      "F-1 score on test data : 0.7702573879885605 \n",
      "recall score on test data : 0.7875243664717348 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABQwUlEQVR4nO3deVxU9f7H8fcMyIDAgKiAlKKmuZRLailt7iuZplaWFZZtZrmlbb9S05uWN/dcsktqpS1WenNJM029Ji7XxMzKq6ZhCWiaLC5sc35/EJPHpQ7TIEy+nvdxHo845zvf8535/YhP7+/3fMdmGIYhAAAAoJjspT0AAAAA+CYKSQAAAHiEQhIAAAAeoZAEAACARygkAQAA4BEKSQAAAHiEQhIAAAAeoZAEAACARygkAQAA4BEKSQBlwp49e9ShQweFhYXJZrNp8eLFXu3/wIEDstlsmjt3rlf79WWtWrVSq1atSnsYAHwYhSQAt3379umRRx5RzZo1FRgYKKfTqRtuuEFTpkzRqVOnSvTeCQkJ2rlzp1566SW9/fbbatasWYne72Lq27evbDabnE7neT/HPXv2yGazyWaz6dVXXy12/4cOHdKoUaOUnJzshdECgHX+pT0AAGXDsmXLdPvtt8vhcOi+++7T1VdfrdzcXG3YsEHDhw/Xrl27NHv27BK596lTp5SUlKT/+7//0+OPP14i94iNjdWpU6dUrly5Eun/z/j7++vkyZNasmSJ7rjjDtO1+fPnKzAwUKdPn/ao70OHDunFF19U9erV1bhxY8uv++yzzzy6HwAUoZAEoP3796t3796KjY3VmjVrVKVKFfe1AQMGaO/evVq2bFmJ3f/IkSOSpPDw8BK7h81mU2BgYIn1/2ccDoduuOEGvfvuu+cUkgsWLFB8fLw++uijizKWkydPqnz58goICLgo9wPw98XUNgCNHz9e2dnZSkxMNBWRRWrVqqVBgwa5f87Pz9eYMWN0xRVXyOFwqHr16nruueeUk5Njel316tV1yy23aMOGDbruuusUGBiomjVr6q233nK3GTVqlGJjYyVJw4cPl81mU/Xq1SUVTgkX/fOZRo0aJZvNZjq3atUq3XjjjQoPD1dISIjq1Kmj5557zn39Qmsk16xZo5tuuknBwcEKDw9Xt27d9N133533fnv37lXfvn0VHh6usLAw3X///Tp58uSFP9iz3H333fr00091/Phx97mtW7dqz549uvvuu89pf+zYMQ0bNkwNGjRQSEiInE6nOnfurB07drjbrF27Vtdee60k6f7773dPkRe9z1atWunqq6/Wtm3bdPPNN6t8+fLuz+XsNZIJCQkKDAw85/137NhRFSpU0KFDhyy/VwCXBgpJAFqyZIlq1qyp66+/3lL7Bx98UCNGjFCTJk00adIktWzZUuPGjVPv3r3Pabt371716tVL7du314QJE1ShQgX17dtXu3btkiT16NFDkyZNkiTdddddevvttzV58uRijX/Xrl265ZZblJOTo9GjR2vChAm69dZb9eWXX/7h6z7//HN17NhRhw8f1qhRozR06FBt3LhRN9xwgw4cOHBO+zvuuENZWVkaN26c7rjjDs2dO1cvvvii5XH26NFDNptNH3/8sfvcggULVLduXTVp0uSc9j/88IMWL16sW265RRMnTtTw4cO1c+dOtWzZ0l3U1atXT6NHj5YkPfzww3r77bf19ttv6+abb3b3c/ToUXXu3FmNGzfW5MmT1bp16/OOb8qUKapcubISEhJUUFAgSXr99df12Wefadq0aYqJibH8XgFcIgwAl7SMjAxDktGtWzdL7ZOTkw1JxoMPPmg6P2zYMEOSsWbNGve52NhYQ5Kxfv1697nDhw8bDofDePLJJ93n9u/fb0gy/vnPf5r6TEhIMGJjY88Zw8iRI40z//U1adIkQ5Jx5MiRC4676B5z5sxxn2vcuLERGRlpHD161H1ux44dht1uN+67775z7vfAAw+Y+rztttuMihUrXvCeZ76P4OBgwzAMo1evXkbbtm0NwzCMgoICIzo62njxxRfP+xmcPn3aKCgoOOd9OBwOY/To0e5zW7duPee9FWnZsqUhyZg1a9Z5r7Vs2dJ0buXKlYYk4x//+Ifxww8/GCEhIUb37t3/9D0CuDSRSAKXuMzMTElSaGiopfbLly+XJA0dOtR0/sknn5Skc9ZS1q9fXzfddJP758qVK6tOnTr64YcfPB7z2YrWVv773/+Wy+Wy9JrU1FQlJyerb9++ioiIcJ9v2LCh2rdv736fZ3r00UdNP9900006evSo+zO04u6779batWuVlpamNWvWKC0t7bzT2lLhukq7vfBf0wUFBTp69Kh72v6rr76yfE+Hw6H777/fUtsOHTrokUce0ejRo9WjRw8FBgbq9ddft3wvAJcWCkngEud0OiVJWVlZltr/+OOPstvtqlWrlul8dHS0wsPD9eOPP5rOV6tW7Zw+KlSooF9//dXDEZ/rzjvv1A033KAHH3xQUVFR6t27tz744IM/LCqLxlmnTp1zrtWrV0+//PKLTpw4YTp/9nupUKGCJBXrvXTp0kWhoaF6//33NX/+fF177bXnfJZFXC6XJk2apNq1a8vhcKhSpUqqXLmyvv76a2VkZFi+52WXXVasB2teffVVRUREKDk5WVOnTlVkZKTl1wK4tFBIApc4p9OpmJgYffPNN8V63dkPu1yIn5/fec8bhuHxPYrW7xUJCgrS+vXr9fnnn+vee+/V119/rTvvvFPt27c/p+1f8VfeSxGHw6EePXpo3rx5WrRo0QXTSEkaO3ashg4dqptvvlnvvPOOVq5cqVWrVumqq66ynLxKhZ9PcWzfvl2HDx+WJO3cubNYrwVwaaGQBKBbbrlF+/btU1JS0p+2jY2Nlcvl0p49e0zn09PTdfz4cfcT2N5QoUIF0xPORc5OPSXJbrerbdu2mjhxor799lu99NJLWrNmjb744ovz9l00zt27d59z7fvvv1elSpUUHBz8197ABdx9993avn27srKyzvuAUpEPP/xQrVu3VmJionr37q0OHTqoXbt253wmVot6K06cOKH7779f9evX18MPP6zx48dr69atXusfwN8LhSQAPfXUUwoODtaDDz6o9PT0c67v27dPU6ZMkVQ4NSvpnCerJ06cKEmKj4/32riuuOIKZWRk6Ouvv3afS01N1aJFi0ztjh07ds5rizbmPntLoiJVqlRR48aNNW/ePFNh9s033+izzz5zv8+S0Lp1a40ZM0avvfaaoqOjL9jOz8/vnLRz4cKF+vnnn03nigre8xXdxfX0008rJSVF8+bN08SJE1W9enUlJCRc8HMEcGljQ3IAuuKKK7RgwQLdeeedqlevnumbbTZu3KiFCxeqb9++kqRGjRopISFBs2fP1vHjx9WyZUtt2bJF8+bNU/fu3S+4tYwnevfuraefflq33XabBg4cqJMnT2rmzJm68sorTQ+bjB49WuvXr1d8fLxiY2N1+PBhzZgxQ5dffrluvPHGC/b/z3/+U507d1ZcXJz69eunU6dOadq0aQoLC9OoUaO89j7OZrfb9fzzz/9pu1tuuUWjR4/W/fffr+uv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9913NXXqVH333Xd6/fXXFRISoqpVq+qjjz6SJO3evVupqamaMmWKJGncuHF66623NGvWLO3atUtDhgzRPffco3Xr1kkqLHh79Oihrl27Kjk5WQ8++KCeeeaZkvrYAKBEMLUNwCcYhqHVq1dr5cqVeuKJJ3TkyBEFBwfrX//6l3tK+5133pHL5dK//vUv2Ww2SdKcOXMUHh6utWvXqkOHDpo8ebKeffZZ9ejRQ5I0a9YsrVy58oL3/d///qcPPvhAq1atUrt27SRJNWvWdF8vmgaPjIxUeHi4pMIEc+zYsfr8888VFxfnfs2GDRv0+uuvq2XLlpo5c6auuOIKTZgwQZJUp04d7dy5U6+88ooXPzUAKFkUkgDKtKVLlyokJER5eXlyuVy6++67NWrUKA0YMEANGjQwrYvcsWOH9u7dq9DQUFMfp0+f1r59+5SRkaHU1FQ1b97cfc3f31/NmjU7Z3q7SHJysvz8/NSyZUvLY967d69Onjyp9u3bm87n5ubqmmuukSR99913pnFIchedAOArKCQBlGmtW7fWzJkzFRAQoJiYGPn7//6vreDgYFPb7OxsNW3aVPPnzz+nn8qVK3t0/6CgoGK/Jjs7W5K0bNkyXXbZZaZrDofDo3EAQFlEIQmgTAsODlatWrUstW3SpInef/99RUZGyul0nrdNlSpVtHnzZt18882SpPz8fG3btk1NmjQ5b/sGDRrI5XJp3bp17qntMxUlogUFBe5z9evXl8PhUEpKygWTzHr16umTTz4xndu0adOfv0kAKEN42AbA30afPn1UqVIldevWTf/5z3+0f/9+rV27VgMHDtRPP/0kSRo0aJBefvllLV68WN9//70ee+yxP9wDsnr16kpISNADDzygxYsXu/v84IMPJEmxsbGy2WxaunSpjhw5ouzsbIWGhmrYsGEaMmSI5s2bp3379umrr77StGnTNG/ePEnSo48+qj179mj48OHavXu3FixYoLlz55b0RwQAXkUhCeBvo3z58lq/fr2qVaumHj16qF69eurXr59Onz7tTiiffPJJ3XvvvUpISFBcXJxCQ0N12223/WG/M2fOVK9evfTYY4+pbt26euihh3TixAlJ0mWXXaYXX3xRzzzzjKKiovT4449LksaMGaMXXnhB48aNU7169dSpUyctW7ZMNWrUkCRVq1ZNH330kRYvXqxGjRpp1qxZGjt2bAl+OgDgfTbjQivMAQAAgD9AIgkAAACPUEgCAADAIxSSAAAA8AiFJAAAADxCIQkAAACPUEgCAADAIxSSAAAA8AiFJAAAADxCIQkAAACPUEgCAADAIxSSAAAA8Mj/A4+t/h1klmkIAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "Report(\"AdaBoostClassifier\", AdaBoostClassifier(), **input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RandomForestClassifier\n",
      "accuracy score on training data : 0.7449867117661271 \n",
      "F-1 score on training data : 0.7522591245159018 \n",
      "recall score on training data : 0.7726615236258437 \n",
      "accuracy score on test data : 0.7468599033816425 \n",
      "F-1 score on test data : 0.7507136060894386 \n",
      "recall score on test data : 0.7690058479532164 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABMiUlEQVR4nO3dd3wU1d7H8e9syiYk2YSaEIVQBaI0wQux0HtUqoqNgNgQpQkiPl5EUPCiKBYU0VAs2IUrRRBBgggookEuKhcQDF6S0CQFTJ/nj5iVoTlZNyQrnzeveT1k5uyZs3ufmB/fc+bEME3TFAAAAFBKjvIeAAAAAHwThSQAAAA8QiEJAAAAj1BIAgAAwCMUkgAAAPAIhSQAAAA8QiEJAAAAj1BIAgAAwCMUkgAAAPAIhSSACmHnzp3q1q2bwsPDZRiGFi9e7NX+9+7dK8MwNH/+fK/268s6dOigDh06lPcwAPgwCkkAbrt379Zdd92levXqKSgoSC6XS1dccYWeffZZ/fbbb2V674SEBG3btk2PP/64Xn/9dbVu3bpM73cuDR48WIZhyOVynfZz3LlzpwzDkGEYeuqpp0rd//79+zVp0iQlJyd7YbQAYJ9/eQ8AQMWwbNkyXXfddXI6nRo0aJAuueQS5eXlaf369Ro3bpy2b9+uOXPmlMm9f/vtN23cuFH/93//p3vvvbdM7hETE6PffvtNAQEBZdL/n/H399fx48e1ZMkSXX/99ZZrb775poKCgpSTk+NR3/v379ejjz6qOnXqqEWLFrZf98knn3h0PwAoQSEJQHv27NHAgQMVExOjNWvWqGbNmu5rw4cP165du7Rs2bIyu//BgwclSREREWV2D8MwFBQUVGb9/xmn06krrrhCb7311imF5MKFCxUfH68PPvjgnIzl+PHjqlSpkgIDA8/J/QD8fTG1DUDTp09Xdna2EhMTLUVkiQYNGmjkyJHurwsKCjRlyhTVr19fTqdTderU0UMPPaTc3FzL6+rUqaOrr75a69ev1z/+8Q8FBQWpXr16eu2119xtJk2apJiYGEnSuHHjZBiG6tSpI6l4Srjk7yeaNGmSDMOwnFu1apWuvPJKRUREKDQ0VI0aNdJDDz3kvn6mNZJr1qzRVVddpZCQEEVERKh379764YcfTnu/Xbt2afDgwYqIiFB4eLiGDBmi48ePn/mDPclNN92kjz/+WEePHnWf27x5s3bu3KmbbrrplPZHjhzR2LFj1bRpU4WGhsrlcqlnz57aunWru83atWt12WWXSZKGDBniniIveZ8dOnTQJZdcoi1btqhdu3aqVKmS+3M5eY1kQkKCgoKCTnn/3bt3V+XKlbV//37b7xXA+YFCEoCWLFmievXq6fLLL7fV/vbbb9fEiRN16aWX6plnnlH79u01bdo0DRw48JS2u3bt0oABA9S1a1fNmDFDlStX1uDBg7V9+3ZJUr9+/fTMM89Ikm688Ua9/vrrmjlzZqnGv337dl199dXKzc3V5MmTNWPGDF177bX64osvzvq6Tz/9VN27d9eBAwc0adIkjRkzRhs2bNAVV1yhvXv3ntL++uuvV1ZWlqZNm6brr79e8+fP16OPPmp7nP369ZNhGPrwww/d5xYuXKjGjRvr0ksvPaX9Tz/9pMWLF+vqq6/W008/rXHjxmnbtm1q3769u6hr0qSJJk+eLEm688479frrr+v1119Xu3bt3P0cPnxYPXv2VIsWLTRz5kx17NjxtON79tlnVb16dSUkJKiwsFCS9PLLL+uTTz7R888/r+joaNvvFcB5wgRwXsvIyDAlmb1797bVPjk52ZRk3n777ZbzY8eONSWZa9ascZ+LiYkxJZnr1q1znztw4IDpdDrN+++/331uz549piTzySeftPSZkJBgxsTEnDKGRx55xDzxP1/PPPOMKck8ePDgGcddco958+a5z7Vo0cKsUaOGefjwYfe5rVu3mg6Hwxw0aNAp97vtttssffbt29esWrXqGe954vsICQkxTdM0BwwYYHbu3Nk0TdMsLCw0o6KizEcfffS0n0FOTo5ZWFh4yvtwOp3m5MmT3ec2b958ynsr0b59e1OSOXv27NNea9++veXcypUrTUnmY489Zv70009maGio2adPnz99jwDOTySSwHkuMzNTkhQWFmar/fLlyyVJY8aMsZy///77JemUtZSxsbG66qqr3F9Xr15djRo10k8//eTxmE9Wsrby3//+t4qKimy9JjU1VcnJyRo8eLCqVKniPt+sWTN17drV/T5PdPfdd1u+vuqqq3T48GH3Z2jHTTfdpLVr1yotLU1r1qxRWlraaae1peJ1lQ5H8X+mCwsLdfjwYfe0/TfffGP7nk6nU0OGDLHVtlu3brrrrrs0efJk9evXT0FBQXr55Zdt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19y5c7V+/XodOXJEixYtOut9Bw0apLfeekvPPfecfvjhB7388ssKDQ1VrVq19MEHH0iSduzYodTUVD377LOSpGnTpum1117T7NmztX37do0ePVq33HKLkpKSJBX/UO3Xr5+uueYaJScn6/bbb9eDDz541nFs2bJF+fn56tKli/tc48aNVbt2bW3cuLH0HyjgJef79ybwt2cCPiYhIcHs3bu3aZqmWVRUZK5atcp0Op3m2LFj3dcjIyPN3Nxc92tef/11s1GjRmZRUZH7XG5urhkcHGyuXLnSNE3TrFmzpjl9+nT39fz8fPPCCy9038s0TbN9+/bmyJEjTdM0zR07dpiSzFWrVp12nJ999pkpyfz111/d53JycsxKlSqZGzZssLQdOnSoeeONN5qmaZoTJkwwY2NjLdfHjx9/Sl8nevPNN83AwMBTzl922WXmAw88cNrXAN7G9+bZne6+gK/zL8caFvDY0qVLFRoaqvz8fBUVFemmm27SpEmT3NebNm2qwMBA99dbt27Vrl27TllDlZOTo927dysjI0Opqalq06aN+5q/v79at259yhRaieTkZPn5+al9+/a2x71r1y4dP35cXbt2tZzPy8tTy5YtJUk//PCDZRySFBcXZ/seQHniexM4v1BIwid17NhRL730kgIDAxUdHS1/f+v/K4eEhFi+zs7OVqtWrfTmm2+e0lf16tU9GkNwcHCpX5OdnS1JWrZsmS644ALLtb/ye4CjoqKUl5eno0ePKiIiwn0+PT1dUVFRHvcLlBbfm8D5hUISPikkJEQNGjSw3f7SSy/VO++8oxo1asjlcp22Tc2aNfXll1+qXbt2kqSCggJt2bJFl1566WnbN23aVEVFRUpKSrKsTSxRkroUFha6z8XGxsrpdColJeWMaUmTJk3cDyeU+LMHZlq1aqWAgACtXr1a/fv3l1S8/islJYXEBOcU35vA+YWHbXBeuPnmm1WtWjX17t1bn3/+ufbs2aO1a9dqxIgR+uWXXyRJI0eO1BNPPKHFixfrxx9/1D333HPW/d7q1KmjhIQE3XbbbVq8eLG7z3fffVeSFBMTI8MwtHTpUh08eFDZ2dkKCwvT2LFjNXr0aC1YsEC7d+/WN998o+eff14LFiyQJN19993auXOnxo0bpx07dmjhwoWaP3/+Wd9feHi4hg4dqjFjxuizzz7Tli1bNGTIEMXFxfGkKCq0v/v3piSlpaUpOTlZu3btkiRt27ZNycnJOnLkyF/78ICKoLwXaQKldeKC/tJcT01NNQcNGmRWq1bNdDqdZr169cw77rjDzMjIME2zeAH/yJEjTZfLZUZERJhjxowxBw0adMYF/aZpmr/99ps5evRos2bNmmZgYKDZoEEDc+7cue7rkydPNqOiokzDMMyEhATTNIsfQpg5c6bZqFEjMyAgwKxevbrZvXt3Mykpyf26JUuWmA0aNDCdTqd51VVXmXPnzv3TRfq//fabec8995iVK1c2K1WqZPbt29dMTU0962cJeBPfm6f3yCOPmJJOOebNm3e2jxPwCYZpnmG1MgAAAHAWTG0DAADAIxSSAAAA8AiFJAAAADxCIQkAAACPUEgCAADAIxSSAAAA8AiFJAAAADxCIQkAAACPUEgCOO8NHjxYffr0cX/doUMHjRo16pyPY+3atTIM46y//g8AKhIKSQAV1uDBg2UYhgzDUGBgoBo0aKDJkyeroKCgTO/74YcfasqUKbbaUvwBOJ/5l/cAAOBsevTooXnz5ik3N1fLly/X8OHDFRAQoAkTJlja5eXlKTAw0Cv3rFKlilf6AYC/OxJJABWa0+lUVFSUYmJiNGzYMHXp0kUfffSRezr68ccfV3R0tBo1aiRJ2rdvn66//npFRESoSpUq6t27t/bu3evur7CwUGPGjFFERISqVq2qBx54QKZpWu558tR2bm6uxo8fr1q1asnpdKpBgwZKTEzU3r171bFjR0lS5cqVZRiGBg8eLEkqKirStGnTVLduXQUHB6t58+Z6//33LfdZvny5LrroIgUHB6tjx46WcQKAL6CQBOBTgoODlZeXJ0lavXq1duzYoVWrVmnp0qXKz89X9+7dFRYWps8//1xffPGFQkND1aNHD/drZsyYofnz52vu3Llav369jhw5okWLFp31noMGDdJbb72l5557Tj/88INefvllhYaGqlatWvrggw8kSTt27FBqaqqeffZZSdK0adP02muvafbs2dq+fbtGjx6tW265RUlJSZKKC95+/frpmmuuUXJysm6//XY9+OCDZfWxAUCZYGobgE8wTVOrV6/WypUrdd999+ngwYMKCQnRq6++6p7SfuONN1RUVKRXX31VhmFIkubNm6eIiAitXbtW3bp108yZMzVhwgT169dPkjR79mytXLnyjPf973//q3fffVerVq1Sly5dJEn16tVzXy+ZBq9Ro4YiIiIkFSeYU6dO1aeffqq4uDj3a9avX6+XX35Z7du310svvaT69etrxowZkqRGjRpp27Zt+te//uXFTw0AyhaFJIAKbenSpQoNDVV+fr6Kiop00003adKkSRo+fLiaNm1qWRe5detW7dq1S2FhYZY+cnJytHv3bmVkZCg1NVVt2rRxX/P391fr1q1Pmd4ukZycLD8/P7Vv3972mHft2qXjx4+ra9eulvN5eXlq2bKlJOmHH36wjEOSu+gEAF9BIQmgQuvYsaNeeuklBQYGKjo6Wv7+f/xnKyQkxNI2OztbrVq10ptvvnlKP9WrV/fo/sHBwaV+TXZ2tiRp2bJluuCCCyzXnE6nR+MAgIqIQhJAhRYSEqIGDRrYanvppZfqnXfeUY0aNeRyuU7bpmbNmvryyy/Vrl07SVJBQYG2bNmiSy+99LTtmzZtqqKiIiUlJbmntk9UkogWFha6z8XGxsrpdColJeWMSWaTJk300UcfWc5t2rTpz98kAFQgPGwD4G/j5ptvVrVq1dS7d299/vnn2rNnj9auXasRI0bol19+kSSNHDlSTzzxhBYvXqwff/xR99xzz1n3gKxTp44SEhJ02223afHixe4+3333XUlSTEyMDMPQ0qVLdfDgQWVnZyssLExjx47V6NGjtWDBAu3evVvffPONnn/+eS1YsECSdPfdd2vnzp0aN26cduzYoYULF2r+/Pll/REBgFdRSAL426hUqZLWrVun2rVrq1+/fmrSpImGDh2qnJwcd0J5//3369Zbb1VCQoLi4uIUFhamvn37nrXfl156SQMGDNA999yjxo0b64477tCxY8ckSRdccIEeffRRPfjgg4qMjNS9994rSZoyZYr++c9/atq0aWrSpIl69OihZcuWqW7dupKk2rVr64MPPtDixYvVvHlzzZ49W1OnTi3DTwcAvM8wz7TCHAAAADgLEkkAAAB4hEISAAAAHqGQBAAAgEcoJAEAAOARCkkAAAB4hEISAAAAHqGQBAAAgEcoJAEAAOARCkkAAAB4hEISAAAAHqGQBAAAgEf+Hx7fg3NBRbmTAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "Report(\"RandomForestClassifier\", RandomForestClassifier(n_estimators = 50, max_depth =2), **input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier\n",
      "accuracy score on training data : 0.996859144720947 \n",
      "F-1 score on training data : 0.9968742486174561 \n",
      "recall score on training data : 0.9995178399228544 \n",
      "accuracy score on test data : 0.9048309178743962 \n",
      "F-1 score on test data : 0.9093419236079153 \n",
      "recall score on test data : 0.9629629629629629 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABLmElEQVR4nO3deZyN5f/H8fc5s5wZY86MscyYYghZStZiWiyRXUTLlGosUSJbJC2IUL4pkUxqrKGdsqR8yUghXxpJJUSjHzO2zGKZ9f79Mc3Jbal7TmfMnLyeHvfjYe77Otd9nfP9jvn0vq77GpthGIYAAACAQrIX9wAAAADgnSgkAQAA4BYKSQAAALiFQhIAAABuoZAEAACAWygkAQAA4BYKSQAAALiFQhIAAABuoZAEAACAWygkAZQIu3fvVps2bRQSEiKbzaalS5d6tP/9+/fLZrNp7ty5Hu3Xm7Vo0UItWrQo7mEA8GIUkgBc9u7dq4cfflhXXXWVAgIC5HQ6ddNNN+nVV1/V6dOni/TesbGx2rFjhyZMmKAFCxaocePGRXq/S6lnz56y2WxyOp0X/Bx3794tm80mm82ml156qdD9Hzx4UGPHjlViYqIHRgsA1vkW9wAAlAwrVqzQXXfdJYfDoQcffFDXXnutsrKytGHDBo0YMUI7d+7UrFmziuTep0+f1saNG/X0009r4MCBRXKPqKgonT59Wn5+fkXS/9/x9fXVqVOntGzZMt19992mawsXLlRAQIDOnDnjVt8HDx7Uc889pypVqqh+/fqWX/f555+7dT8AKEAhCUD79u1TTEyMoqKitHbtWlWsWNF1bcCAAdqzZ49WrFhRZPc/cuSIJCk0NLTI7mGz2RQQEFBk/f8dh8Ohm266SYsXLz6vkFy0aJE6duyoDz/88JKM5dSpUypVqpT8/f0vyf0A/HsxtQ1AkydPVkZGhuLj401FZIHq1atr8ODBrq9zcnI0fvx4VatWTQ6HQ1WqVNFTTz2lzMxM0+uqVKmiTp06acOGDbrhhhsUEBCgq666SvPnz3e1GTt2rKKioiRJI0aMkM1mU5UqVSTlTwkX/P1sY8eOlc1mM51bvXq1br75ZoWGhqp06dKqWbOmnnrqKdf1i62RXLt2rW655RYFBQUpNDRUXbp00Y8//njB++3Zs0c9e/ZUaGioQkJC1KtXL506deriH+w57rvvPn366ac6ceKE69yWLVu0e/du3Xfffee1P378uIYPH666deuqdOnScjqdat++vbZv3+5qs27dOl1//fWSpF69ermmyAveZ4sWLXTttddq69atatasmUqVKuX6XM5dIxkbG6uAgIDz3n/btm1VpkwZHTx40PJ7BXB5oJAEoGXLlumqq67SjTfeaKn9Qw89pNGjR6thw4Z65ZVX1Lx5c02aNEkxMTHntd2zZ4/uvPNO3XbbbZoyZYrKlCmjnj17aufOnZKkbt266ZVXXpEk3XvvvVqwYIGmTp1aqPHv3LlTnTp1UmZmpsaNG6cpU6bo9ttv11dfffWXr/vvf/+rtm3b6vDhwxo7dqyGDRumr7/+WjfddJP2799/Xvu7775b6enpmjRpku6++27NnTtXzz33nOVxduvWTTabTR999JHr3KJFi1SrVi01bNjwvPa//PKLli5dqk6dOunll1/WiBEjtGPHDjVv3txV1NWuXVvjxo2TJPXr108LFizQggUL1KxZM1c/x44dU/v27VW/fn1NnTpVLVu2vOD4Xn31VZUvX16xsbHKzc2VJL3xxhv6/PPPNX36dEVGRlp+rwAuEwaAy1pqaqohyejSpYul9omJiYYk46GHHjKdHz58uCHJWLt2retcVFSUIclYv36969zhw4cNh8NhPP74465z+/btMyQZ//nPf0x9xsbGGlFRUeeNYcyYMcbZ/3y98sorhiTjyJEjFx13wT3mzJnjOle/fn2jQoUKxrFjx1zntm/fbtjtduPBBx887369e/c29XnHHXcYZcuWveg9z34fQUFBhmEYxp133mm0atXKMAzDyM3NNSIiIoznnnvugp/BmTNnjNzc3PPeh8PhMMaNG+c6t2XLlvPeW4HmzZsbkoy4uLgLXmvevLnp3GeffWZIMp5//nnjl19+MUqXLm107dr1b98jgMsTiSRwmUtLS5MkBQcHW2q/cuVKSdKwYcNM5x9//HFJOm8tZZ06dXTLLbe4vi5fvrxq1qypX375xe0xn6tgbeXHH3+svLw8S685dOiQEhMT1bNnT4WFhbnOX3fddbrttttc7/NsjzzyiOnrW265RceOHXN9hlbcd999WrdunZKTk7V27VolJydfcFpbyl9Xabfn/zOdm5urY8eOuabtt23bZvmeDodDvXr1stS2TZs2evjhhzVu3Dh169ZNAQEBeuONNyzfC8DlhUISuMw5nU5JUnp6uqX2v/76q+x2u6pXr246HxERodDQUP3666+m85UrVz6vjzJlyuj33393c8Tnu+eee3TTTTfpoYceUnh4uGJiYvTee+/9ZVFZMM6aNWued6127do6evSoTp48aTp/7nspU6aMJBXqvXTo0EHBwcF69913tXDhQl1//fXnfZYF8vLy9Morr6hGjRpyOBwqV66cypcvr++++06pqamW73nFFVcU6sGal156SWFhYUpMTNS0adNUoUIFy68FcHmhkAQuc06nU5GRkfr+++8L9bpzH3a5GB8fnwueNwzD7XsUrN8rEBgYqPXr1+u///2vHnjgAX333Xe65557dNttt53X9p/4J++lgMPhULdu3TRv3jwtWbLkommkJE2cOFHDhg1Ts2bN9Pbbb+uzzz7T6tWrdc0111hOXqX8z6cwvv32Wx0+fFiStGPHjkK9FsDlhUISgDp16qS9e/dq48aNf9s2KipKeXl52r17t+l8SkqKTpw44XoC2xPKlCljesK5wLmppyTZ7Xa1atVKL7/8sn744QdNmDBBa9eu1RdffHHBvgvGuWvXrvOu/fTTTypXrpyCgoL+2Ru4iPvuu0/ffvut0tPTL/iAUoEPPvhALVu2VHx8vGJiYtSmTRu1bt36vM/EalFvxcmTJ9WrVy/VqVNH/fr10+TJk7VlyxaP9Q/g34VCEoCeeOIJBQUF6aGHHlJKSsp51/fu3atXX31VUv7UrKTznqx++eWXJUkdO3b02LiqVaum1NRUfffdd65zhw4d0pIlS0ztjh8/ft5rCzbmPndLogIVK1ZU/fr1NW/ePFNh9v333+vzzz93vc+i0LJlS40fP16vvfaaIiIiLtrOx8fnvLTz/fff1//93/+ZzhUUvBcqugtr5MiRSkpK0rx58/Tyyy+rSpUqio2NvejnCODyxobkAFStWjUtWrRI99xzj2rXrm36zTZff/213n//ffXs2VOSVK9ePcXGxmrWrFk6ceKEmjdvrm+++Ubz5s1T165dL7q1jDtiYmI0cuRI3XHHHRo0aJBOnTqlmTNn6uqrrzY9bDJu3DitX79eHTt2VFRUlA4fPqzXX39dV155pW6++eaL9v+f//xH7du3V3R0tPr06aPTp09r+vTpCgkJ0dixYz32Ps5lt9v1zDPP/G27Tp06ady4cerVq5duvPFG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+Ojw4cPnnT927Jh8fHw8MigAAICSiF+RaFboRNIwjAuez8zMlL+//z8eEAAAQElls0l2tv9xsVxITps2TVJ+Jf7WW2+pdOnSrmu5ublav349ayQBAAAuI5YLyVdeeUVSfiIZFxdnmsb29/dXlSpVFBcX5/kRAgAAlBD2IkgkPd3fpWS5kNy3b58kqWXLlvroo49UpkyZIhsUAAAASr5Cr5H84osvimIcAAAAJR4bkpsV+qnt7t2768UXXzzv/OTJk3XXXXd5ZFAAAAAo+QpdSK5fv14dOnQ473z79u21fv16jwwKAACgJCpYI+npw1sVupDMyMi44DY/fn5+SktL88igAAAAUPIVupCsW7eu3n333fPOv/POO6pTp45HBgUAAFAS2WxFc3irQj9s8+yzz6pbt27au3evbr31VknSmjVrtGjRIn3wwQceHyAAAEBJYbfZZPdw5efp/i6lQheSnTt31tKlSzVx4kR98MEHCgwMVL169bR27VqFhYUVxRgBAABQAhW6kJSkjh07qmPHjpKktLQ0LV68WMOHD9fWrVuVm5vr0QECAACUFHa5sS7QQp/eyu2xr1+/XrGxsYqMjNSUKVN06623atOmTZ4cGwAAAEqwQiWSycnJmjt3ruLj45WWlqa7775bmZmZWrp0KQ/aAACAf72ieDjGi5dIWk8kO3furJo1a+q7777T1KlTdfDgQU2fPr0oxwYAAIASzHIi+emnn2rQoEHq37+/atSoUZRjAgAAKJHsKoKntuW9kaTlRHLDhg1KT09Xo0aN1KRJE7322ms6evRoUY4NAAAAJZjlQrJp06Z68803dejQIT388MN65513FBkZqby8PK1evVrp6elFOU4AAIBix4bkZoV+ajsoKEi9e/fWhg0btGPHDj3++ON64YUXVKFCBd1+++1FMUYAAIASgd+1bfaPti6qWbOmJk+erN9++02LFy/21JgAAADgBdzakPxcPj4+6tq1q7p27eqJ7gAAAEokm83zv9LwspraBgAAACQPJZIAAACXAzYkNyORBAAAgFtIJAEAACwqiqesL9untgEAAHD5IpEEAACwyPbHH0/36a0oJAEAACxiatuMqW0AAAC4hUQSAADAIhJJMxJJAAAAuIVEEgAAwCKbzSabx39FovdGkiSSAAAAcAuJJAAAgEWskTQjkQQAAIBbSCQBAAAsstny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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier\n",
      "accuracy score on training data : 0.6371104131432713 \n",
      "F-1 score on training data : 0.5156401160915833 \n",
      "recall score on training data : 0.38548698167791706 \n",
      "accuracy score on test data : 0.6584541062801932 \n",
      "F-1 score on test data : 0.5394136807817589 \n",
      "recall score on test data : 0.40350877192982454 \n"
     ]
    },
    {
     "data": {
      "image/png": "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goKC9PLLL1t+FoDzC4kkcJ5zOp2SpCNHjlhq/8MPP8hut6tJkyam89HR0QoPD9cPP/xgOt+gQYNT+qhdu7Z++eUXLyM+1a233qqrrrpK99xzj6KiojRgwAC9+eabZ00qK+Js1qzZKddatGihn376SUePHjWdP/m91K5dW5I8ei+9e/dWWFiY3njjDS1ZskSXX375KZ9lhfLycs2cOVNNmzaVw+FQ3bp1Va9ePX3xxRfKy8uz/MwLLrjAo4U1zz77rCIiIpSenq7Zs2crMjLS8r0Azi8kksB5zul0KiYmRl9++aVH95282OVM/Pz8TnveMAyvn1Exf69CcHCwNm/erA8//FB33nmnvvjiC9166626/vrrT2n7e/ye91LB4XCof//+WrRokd59990zViMlaerUqRo9erQ6d+6s1157TWvXrtW6det0ySWXWK68Sic+H0989tlnysnJkSTt2rXLo3sBnF9IJAHohhtu0Lfffqu0tLTfbBsXF6fy8nLt3bvXdD47O1u5ubnuFdi+ULt2bdMK5wonVz0lyW6367rrrtNzzz2nr776SlOmTNGGDRv00Ucfnbbvijj37NlzyrV///vfqlu3rkJCQn7fGziD22+/XZ999pmOHDly2gVKFd566y1169ZNqampGjBggLp3766EhIRTPhOrSb0VR48e1d13362WLVtq2LBhmj59unbs2OGz/gH8sZBIAtDDDz+skJAQ3XPPPcrOzj7l+rfffqvnn39e0omhWUmnrKx+7rnnJEmJiYk+i6tx48bKy8vTF1984T6XmZmpd99919Tu8OHDp9xbsTH3yVsSVahfv77atGmjRYsWmRKzL7/8Uh988IH7fVaGbt26afLkyXrxxRcVHR19xnZ+fn6nVDuXLVum//znP6ZzFQnv6ZJuTz3yyCPKyMjQokWL9Nxzz6lhw4ZKTk4+4+cI4PzGhuQA1LhxYy1dulS33nqrWrRoYfpmm61bt2rZsmUaNGiQJKl169ZKTk7W/PnzlZubqy5duuiTTz7RokWL1K9fvzNuLeONAQMG6JFHHtFNN92kBx54QMeOHdNLL72kiy++2LTYZNKkSdq8ebMSExMVFxennJwczZ07VxdeeKGuvvrqM/b/l7/8Rb169VJ8fLyGDBmi48eP64UXXpDL5dLEiRN99j5OZrfbNW7cuN9sd8MNN2jSpEm6++67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qVOnTkpLS1OrVq0UFRXlbtOjRw8NHz5cu3fvVtu2bS3HXmWJZGpqqhISEk5JIqUTieT06dP1xRdfqGvXrlq2bJkmT56sp59+Wk6nU507d3a3nTRpku699141btxYRUVFMgxDLVq00Ny5czV16lRNnjxZSUlJGjNmjObPn296/rBhw9SuXTvFxsZq6tSpGjNmjEfv4V//+pe6devmfj169GhJJ4bNT7eNEWqeT37I1Sc/5J7x+od7fjrrwptdB4+o10vbKyEyAL+2Zo95J4T0g3lKP5h3htanKigu04IdlTdHHrAiNjbW9HrChAmaOHHiadvu2rVL8fHxKiwsVGhoqN599121bNlS6enpCgwMVHh4uKl9VFSUsrKyJElZWVmmJLLiesU1T1RZIrlixYozXrviiitMw9L9+/c/44bfnTp10ueff37K+fvuu0/33Xef6dzjjz/u/nvbtm21Y8cO0/Wbb77Z9PrkeZYn69q162+2AQAAfyCV8BWJFQXOAwcOmEZPz1aNbNasmdLT05WXl6e33npLycnJ2rRpk48D+21Vvv0PAABATVGZa20qVmFbERgYqCZNmkiS2rdvrx07duj555/XrbfequLiYuXm5pqqktnZ2e6tEqOjo/XJJ5+Y+qtY1e3pdopVviE5AAAAfp/y8nIVFRWpffv2CggIMC1U3rNnjzIyMhQfHy9Jio+P165du5ST878pIevWrZPT6VTLli09ei4VSQAAAKuqwfY/jz32mHr16qUGDRroyJEjWrp0qTZu3Ki1a9fK5XJpyJAhGj16tCIiIuR0OnX//fcrPj5enTp1kiR1795dLVu21J133qnp06crKytL48aNU0pKisfbFJJIAgAA1CA5OTm66667lJmZKZfLpcsuu0xr167V9ddfL0maOXOm7Ha7kpKSVFRUpB49emju3Lnu+/38/LRy5UoNHz5c8fHxCgkJUXJysiZNmuRxLCSSAAAAFlXm9j9W/fobAE8nKChIc+bM0Zw5c87YJi4uTqtXr/bouafDHEkAAAB4hYokAACARbZK2P7H59sJnUNUJAEAAOAVKpIAAAAWVYNF29UKiSQAAIBVZJImDG0DAADAK1QkAQAALKoO2/9UJ1QkAQAA4BUqkgAAABax/Y8ZFUkAAAB4hYokAACARSzaNqMiCQAAAK9QkQQAALCKkqQJiSQAAIBFbP9j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TdOrr76qefPmaffu3Ro1apTuuOMObdq0SdKJhLd///7q06eP0tPTdc899+jRRx+trI8NACoFQ9sAagTDMLR+/XqtXbtW999/vw4dOqSQkBD97W9/cw9pv/baayovL9ff/vY32Ww2SdKCBQsUHh6ujRs3qnv37po1a5Yee+wx9e/fX5I0b948rV279ozP/eabb/Tmm29q3bp1SkhIkCRddNFF7usVw+CRkZEKDw+XdKKCOXXqVH344YeKj49337Nlyxa9/PLL6tKli1566SU1btxYM2bMkCQ1a9ZMu3bt0jPPPOPDTw0AKheJJIBqbeXKlQoNDVVJSYnKy8t1++23a+LEiUpJSVGrVq1M8yI///xz7du3T2FhYaY+CgsL9e233yovL0+ZmZnq2LGj+5q/v786dOhwyvB2hfT0dPn5+alLly6WY963b5+OHTum66+/3nS+uLhYbdu2lSR9/fXXpjgkuZNOAKgpSCQBVGvdunXTSy+9pMDAQMXExMjf/3//2QoJCTG1LSgoUPv27bVkyZJT+qlXr55Xzw8ODvb4noKCAknSqlWrdMEFF5iuORwOr+IAgOqIRBJAtRYSEqImTZpYatuuXTu98cYbioyMlNPpPG2b+vXra/v27ercubMkqbS0VDt37lS7du1O275Vq1YqLy/Xpk2b3EPbv1ZRES0rK3Ofa9mypRwOhzIyMs5YyWzRooX+8Y9/mM5t27btt98kAFQjLLYB8IcxcOBA1a1bV3379tU///lP7d+/Xxs3btQDDzygH3/8UZL04IMP6umnn9by5cv173//W3/+85/Pugdkw4YNlZycrMGDB2v58uXuPt98801JUlxcnGw2m1auXKlDhw6poKBAYWFhGjNmjEaNGqVFixbp22+/1aeffqoXXnhBixYtkiTdd9992rt3r8aOHas9e/Zo6dKlWrhwYWV/RADgUySSAP4watWqpc2bN6tBgwbq37+/WrRooSFDhqiwsNBdoXzooYd05513Kjk5WfHx8QoLC9NNN9101n5feukl3Xzzzfrzn/+s5s2ba+jQoTp69Kgk6YILLtCTTz6pRx99VFFRURoxYoQkafLkyXriiSc0bdo0tWjRQj179tSqVavUqFEjSVKDBg309ttva/ny5WrdurXmzZunqVOnVuKnAwC+ZzPONMMcAAAAOAsqkgAAAPAKiSQAAAC8QiIJAAAAr5BIAgAAwCskkgAAAPAKiSQAAAC8QiIJAAAAr5BIAgAAwCskkgAAAPAKiSQAAAC8QiIJAAAAr/w/Xo8LRZOts+QAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier\n",
      "accuracy score on training data : 0.7574293307562213 \n",
      "F-1 score on training data : 0.762254321572342 \n",
      "recall score on training data : 0.776036644165863 \n",
      "accuracy score on test data : 0.7608695652173914 \n",
      "F-1 score on test data : 0.7630445189085687 \n",
      "recall score on test data : 0.776803118908382 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABMvklEQVR4nO3deXQUVd7G8ac6Syck6YSwJEQgoCCLA6KAkHFhXwOC4ChuBMUNUVkEUcdRhBEcFEUURTQsLrgDrwKKCAOoBGHQIKJmAMHgQMImWYCsXe8fMS1FAlbaDknL9+Opc+iq27du90zTP55bddswTdMUAAAAUEGOqh4AAAAA/BOFJAAAALxCIQkAAACvUEgCAADAKxSSAAAA8AqFJAAAALxCIQkAAACvUEgCAADAKxSSAAAA8AqFJIBqYfv27erZs6ciIyNlGIaWLFni0/53794twzA0f/58n/brzzp37qzOnTtX9TAA+DEKSQAeO3fu1B133KFzzz1XISEhcrlcuvTSS/Xss8/q+PHjlXrupKQkbd26VY8//rhee+01tWvXrlLPdyYNGzZMhmHI5XKV+z5u375dhmHIMAw99dRTFe5/7969mjhxolJTU30wWgCwL7CqBwCgeli2bJn+9re/yel0aujQofrLX/6igoICff755xo/fry2bdumOXPmVMq5jx8/rpSUFP3973/X3XffXSnniI+P1/HjxxUUFFQp/f+ewMBAHTt2TB9++KGuueYay7E33nhDISEhysvL86rvvXv36rHHHlOjRo3Upk0b28/75JNPvDofAJSikASgXbt2aciQIYqPj9fq1atVr149z7GRI0dqx44dWrZsWaWd/8CBA5KkqKioSjuHYRgKCQmptP5/j9Pp1KWXXqo333yzTCG5cOFCJSYm6v333z8jYzl27Jhq1Kih4ODgM3I+AH9eTG0D0LRp05Sbm6vk5GRLEVmqSZMmGjVqlOdxUVGRJk+erPPOO09Op1ONGjXSQw89pPz8fMvzGjVqpH79+unzzz/XJZdcopCQEJ177rl69dVXPW0mTpyo+Ph4SdL48eNlGIYaNWokqWRKuPTPJ5o4caIMw7DsW7lypS677DJFRUUpPDxczZo100MPPeQ5fqprJFevXq3LL79cYWFhioqK0oABA/T999+Xe74dO3Zo2LBhioqKUmRkpG6++WYdO3bs1G/sSa6//np99NFHOnLkiGffpk2btH37dl1//fVl2h8+fFjjxo1Tq1atFB4eLpfLpT59+mjLli2eNmvWrFH79u0lSTfffLNnirz0dXbu3Fl/+ctftHnzZl1xxRWqUaOG5305+RrJpKQkhYSElHn9vXr1Us2aNbV3717brxXA2YFCEoA+/PBDnXvuufrrX/9qq/2tt96qRx55RBdffLGeeeYZderUSVOnTtWQIUPKtN2xY4euvvpq9ejRQ9OnT1fNmjU1bNgwbdu2TZI0aNAgPfPMM5Kk6667Tq+99ppmzJhRofFv27ZN/fr1U35+viZNmqTp06fryiuv1BdffHHa53366afq1auX9u/fr4kTJ2rs2LFav369Lr30Uu3evbtM+2uuuUY5OTmaOnWqrrnmGs2fP1+PPfaY7XEOGjRIhmFo0aJFnn0LFy5U8+bNdfHFF5dp/+OPP2rJkiXq16+fnn76aY0fP15bt25Vp06dPEVdixYtNGnSJEnS7bffrtdee02vvfaarrjiCk8/hw4dUp8+fdSmTRvNmDFDXbp0KXd8zz77rOrUqaOkpCQVFxdLkl566SV98skneu655xQXF2f7tQI4S5gAzmpZWVmmJHPAgAG22qemppqSzFtvvdWyf9y4caYkc/Xq1Z598fHxpiRz3bp1nn379+83nU6ned9993n27dq1y5RkPvnkk5Y+k5KSzPj4+DJjePTRR80T//p65plnTEnmgQMHTjnu0nPMmzfPs69NmzZm3bp1zUOHDnn2bdmyxXQ4HObQoUPLnO+WW26x9HnVVVeZtWrVOuU5T3wdYWFhpmma5tVXX21269bNNE3TLC4uNmNjY83HHnus3PcgLy/PLC4uLvM6nE6nOWnSJM++TZs2lXltpTp16mRKMmfPnl3usU6dOln2rVixwpRk/vOf/zR//PFHMzw83Bw4cODvvkYAZycSSeAsl52dLUmKiIiw1X758uWSpLFjx1r233fffZJU5lrKli1b6vLLL/c8rlOnjpo1a6Yff/zR6zGfrPTayv/7v/+T2+229Zx9+/YpNTVVw4YNU3R0tGd/69at1aNHD8/rPNGdd95peXz55Zfr0KFDnvfQjuuvv15r1qxRRkaGVq9erYyMjHKntaWS6yodjpK/pouLi3Xo0CHPtP1XX31l+5xOp1M333yzrbY9e/bUHXfcoUmTJmnQoEEKCQnRSy+9ZPtcAM4uFJLAWc7lckmScnJybLX/6aef5HA41KRJE8v+2NhYRUVF6aeffrLsb9iwYZk+atasqV9++cXLEZd17bXX6tJLL9Wtt96qmJgYDRkyRO+8885pi8rScTZr1qzMsRYtWujgwYM6evSoZf/Jr6VmzZqSVKHX0rdvX0VEROjtt9/WG2+8ofbt25d5L0u53W4988wzatq0qZxOp2rXrq06derom2++UVZWlu1znnPOORW6seapp55SdHS0UlNTNXPmTNWtW9f2cwGcXSgkgbOcy+VSXFycvv322wo97+SbXU4lICCg3P2maXp9jtLr90qFhoZq3bp1+vTTT3XTTTfpm2++0bXXXqsePXqUaftH/JHXUsrpdGrQoEFasGCBFi9efMo0UpKmTJmisWPH6oorrtDrr7+uFStWaOXKlbrgggtsJ69SyftTEV9//bX2798vSdq6dWuFngvg7EIhCUD9+vXTzp07lZKS8rtt4+Pj5Xa7tX37dsv+zMxMHTlyxHMHti/UrFnTcodzqZNTT0lyOBzq1q2bnn76aX333Xd6/PHHtXr1av373/8ut+/ScaalpZU59sMPP6h27doKCwv7Yy/gFK6//np9/fXXysnJKfcGpVLvvfeeunTpouTkZA0ZMkQ9e/ZU9+7dy7wndot6O44ePaqbb75ZLVu21O23365p06Zp06ZNPusfwJ8LhSQA3X///QoLC9Ott96qzMzMMsd37typZ599VlLJ1KykMndWP/3005KkxMREn43rvPPOU1ZWlr755hvPvn379mnx4sWWdocPHy7z3NKFuU9ekqhUvXr11KZNGy1YsMBSmH377bf65JNPPK+zMnTp0kWTJ0/W888/r9jY2FO2CwgIKJN2vvvuu/rf//5n2Vda8JZXdFfUhAkTlJ6ergULFujpp59Wo0aNlJSUdMr3EcDZjQXJAei8887TwoULde2116pFixaWX7ZZv3693n33XQ0bNkySdOGFFyopKUlz5szRkSNH1KlTJ23cuFELFizQwIEDT7m0jDeGDBmiCRMm6KqrrtK9996rY8eO6cUXX9T5559vudlk0qRJWrdunRITExUfH6/9+/frhRdeUP369XXZZZedsv8nn3xSffr0UUJCgoYPH67jx4/rueeeU2RkpCZOnOiz13Eyh8Ohhx9++Hfb9evXT5MmTdLNN9+s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wCskkgAAADZxjaSVrULygw8+sN3hlVde6fVgAAAA4D9sFZIDBw601ZlhGCouLv4j4wEAAKi2WEfSylYh6Xa7K3scAAAA1R5T21bcbAMAAACveHWzzdGjR7V27Vqlp6eroKDAcuzee+/1ycAAAACqG+PXzdd9+qsKF5Jff/21+vbtq2PHjuno0aOKjo7WwYMHVaNGDdWtW5dCEgAA4CxR4antMWPGqH///vrll18UGhqqDRs26KefflLbtm311FNPVcYYAQAAqgWHYVTK5q8qXEimpqbqvvvuk8PhUEBAgPLz89WgQQNNmzZNDz30UGWMEQAAANVQhQvJoKAgORwlT6tbt67S09MlSZGRkdqzZ49vRwcAAFCNGEblbP6qwtdIXnTRRdq0aZOaNm2qTp066ZFHHtHBgwf12muv6S9/+UtljBEAAADVUIUTySlTpqhevXqSpMcff1w1a9bUiBEjdODAAc2ZM8fnAwQAAKguSteR9PXmryqcSLZr187z57p16+rjjz/26YAAAADgH7xaRxIAAOBsVBnXNPpxIFnxQrJx48anjWB//PHHPzQgAACA6qoyluvx5+V/KlxIjh492vK4sLBQX3/9tT7++GONHz/eV+MCAABANVfhQnLUqFHl7p81a5b+85///OEBAQAAVFdMbVtV+K7tU+nTp4/ef/99X3UHAACAas5nN9u89957io6O9lV3AAAA1U5lLNdzVi3/c9FFF1lesGmaysjI0IEDB/TCCy/4dHBni/QVj8nlclX1MACcoGb7u6t6CABOYhYXVPUQcJIKF5IDBgywFJIOh0N16tRR586d1bx5c58ODgAAoDpxyIfXBZ7Qp7+qcCE5ceLEShgGAAAA/E2Fi+CAgADt37+/zP5Dhw4pICDAJ4MCAACojviJRKsKJ5KmaZa7Pz8/X8HBwX94QAAAANWVYUgOlv/xsF1Izpw5U1JJJf7KK68oPDzcc6y4uFjr1q3jGkkAAICziO1C8plnnpFUkkjOnj3bMo0dHBysRo0aafbs2b4fIQAAQDXhqIRE0tf9nUm2r5HctWuXdu3apU6dOmnLli2ex7t27VJaWppWrFihDh06VOZYAQAAznoTJ04sc43libPCeXl5GjlypGrVqqXw8HANHjxYmZmZlj7S09OVmJioGjVqqG7duho/fryKiooqPJYKXyP573//u8InAQAA+DOoLguSX3DBBfr00089jwMDfyvpxowZo2XLlundd99VZGSk7r77bg0aNEhffPGFpJJLEhMTExUbG6v169dr3759Gjp0qIKCgjRlypQKjaPCd20PHjxY//rXv8rsnzZtmv72t79VtDsAAABUUGBgoGJjYz1b7dq1JUlZWVlKTk7W008/ra5du6pt27aaN2+e1q9frw0bNkiSPvnkE3333Xd6/fXX1aZNG/Xp00eTJ0/WrFmzVFBQsUXfK1xIrlu3Tn379i2zv0+fPlq3bl1FuwMAAPAbpddI+nqTpOzsbMuWn59/ynFs375dcXFxOvfcc3XDDTcoPT1dkrR582YVFhaqe/funrbNmzdXw4YNlZKSIklKSUlRq1atFBMT42nTq1cvZWdna9u2bRV7PyrUWlJubm65y/wEBQUpOzu7ot0BAABAUoMGDRQZGenZpk6dWm67Dh06aP78+fr444/14osvateuXbr88suVk5OjjIwMBQcHKyoqyvKcmJgYZWRkSJIyMjIsRWTp8dJjFVHhayRbtWqlt99+W4888ohl/1tvvaWWLVtWtDsAAAC/YRi+X/extL89e/bI5XJ59judznLb9+nTx/Pn1q1bq0OHDoqPj9c777yj0NBQ3w7ud1S4kPzHP/6hQYMGaefOnerataskadWqVVq4cKHee+89nw8QAACgunAYhhw+riRL+3O5XJZC0q6oqCidf/752rFjh3r06KGCggIdOXLEkkpmZmYqNjZWkhQbG6uNGzda+ii9q7u0je2xV3Sw/fv315IlS7Rjxw7ddddduu+++/S///1Pq1evVpMmTSraHQAAAP6A3Nxc7dy5U/Xq1VPbtm0VFBSkVatWeY6npaUpPT1dCQkJkqSEhARt3brV8pPXK1eulMvlqvDscoUTSUlKTExUYmKipJILQ998802NGzdOmzdvVnFxsTddAgAAVHsOeZHC2eizIsaNG6f+/fsrPj5ee/fu1aOPPqqAgABdd911ioyM1PDhwzV27FhFR0fL5XLpnnvuUUJCgjp27ChJ6tmzp1q2bKmbbrpJ06ZNU0ZGhh5++GGNHDnylNPpp+JVISmV3L2dnJys999/X3FxcRo0aJBmzZrlbXcAAACw4eeff9Z1112nQ4cOqU6dOrrsssu0YcMG1alTR1LJrxE6HA4NHjxY+fn56tWrl1544QXP8wMCArR0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/vrrcrvdeuWVV2QYhiRp3rx5ioqK0po1a9SzZ0/NmDFDDz74oAYNGiRJmj17tlasWHHK8/73v//VO++8o5UrV6p79+6SpHPPPddzvHQavG7duoqKipJUkmBOmTJFn376qRISEjzP+fzzz/XSSy+pU6dOevHFF3Xeeedp+vTpkqRmzZpp69at+te//uXDdw0AKheFJIBqbenSpQoPD1dhYaHcbreuv/56TZw4USNHjlSrVq0s10Vu2bJFO3bsUEREhKWPvLw87dy5U1lZWdq3b586dOjgORYYGKh27dqVmd4ulZqaqoCAAHXq1Mn2mHfs2KFjx46pR48elv0FBQW66KKLJEnff/+9ZRySPEUnAPgLCkkA1VqXLl304osvKjg4WHFxcQoM/O2vrbCwMEvb3NxctW3bVm+88UaZfurUqePV+UNDQyv8nNzcXEnSsmXLdM4551iOOZ1Or8YBANURhSSAai0sLExNmjSx1fbiiy/W22+/rbp168rlcpXbpl69evryyy91xRVXSJKKioq0efNmXXzxxeW2b9Wqldxut9auXeuZ2j5RaSJaXFzs2deyZUs5nU6lp6efMsls0aKFPvjgA8u+DRs2/P6LBIBqhJttAPxp3HDDDapdu7YGDBigzz77TLt27dKaNWt077336ueff5YkjRo1Sk888YSWLFmiH374QXfddddp14Bs1KiRkpKSdMstt2jJkiWePt955x1JUnx8vAzD0NKlS3XgwAHl5uYqIiJC48aN05gxY7RgwQLt3LlTX331lZ577jktWLBAknTnnXdq+/btGj9+vNLS0rRw4ULNnz+/st8iAPApCkkAfxo1atTQunXr1LBhQw0aNEgtWrTQ8OHDlZeX50ko77vvPt10001KSkpSQkKCIiIidNVVV5223xdffFFXX3217rrrLjVv3ly33Xabjh49Kkk655xz9Nhjj+mBBx5QTEyM7r77bknS5MmT9Y9//ENTp05VixYt1Lt3by1btkyNGzeWJDVs2FDvv/++lixZogsvvFCzZ8/WlClTKvHdAQDfM8xTXWEOAAAAnAaJJAAAALxCIQkAAACvUEgCAADAKxSSAAAA8AqFJAAAALxCIQkAAACvUEgCAADAKxSSAAAA8AqFJAAAALxCIQkAAACvUEgCAADAK/8P/Jm7kU7tfVgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier\n",
      "accuracy score on training data : 0.7515100265764677 \n",
      "F-1 score on training data : 0.7527347036903475 \n",
      "recall score on training data : 0.7548216007714561 \n",
      "accuracy score on test data : 0.763768115942029 \n",
      "F-1 score on test data : 0.76410998552822 \n",
      "recall score on test data : 0.7719298245614035 \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApIAAAIjCAYAAACwHvu2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABMt0lEQVR4nO3dd3wU1f7G8Wc2ZROSbEIoCblAqFKUIqCQq9KlRaWqWANiQ1SKIOJPEYICoigWENEI2CtwpSgiSBAB5aJRROUCiuFekoAgKZTU+f0RszIEcLJsSFY+b1/zumTmzMzZvcb98pwzZw3TNE0BAAAAZeSo6A4AAADAN1FIAgAAwCMUkgAAAPAIhSQAAAA8QiEJAAAAj1BIAgAAwCMUkgAAAPAIhSQAAAA8QiEJAAAAj1BIAqgUduzYoR49eig8PFyGYWjJkiVevf7u3btlGIYWLFjg1ev6ss6dO6tz584V3Q0APoxCEoDbrl27dMcdd6hBgwYKCgqSy+XSJZdcomeeeUZHjx4t13snJCRo69ateuyxx/Taa6+pXbt25Xq/s2nIkCEyDEMul+uk7+OOHTtkGIYMw9CTTz5Z5uvv3btXkyZNUkpKihd6CwD2+Vd0BwBUDsuXL9fVV18tp9Opm2++WRdccIHy8vK0fv16jRs3Ttu2bdO8efPK5d5Hjx7Vxo0b9X//93+6++67y+UesbGxOnr0qAICAsrl+n/F399fR44c0dKlS3XNNddYjr3xxhsKCgrSsWPHPLr23r17NXnyZNWrV0+tW7e2fd4nn3zi0f0AoASFJAD98ssvGjx4sGJjY7VmzRrVqlXLfWzEiBHauXOnli9fXm73379/vyQpIiKi3O5hGIaCgoLK7fp/xel06pJLLtFbb71VqpB88803FR8frw8++OCs9OXIkSOqUqWKAgMDz8r9APx9MbQNQDNmzFBOTo6SkpIsRWSJRo0aaeTIke6fCwoKNGXKFDVs2FBOp1P16tXTgw8+qNzcXMt59erV0xVXXKH169fr4osvVlBQkBo0aKBXX33V3WbSpEmKjY2VJI0bN06GYahevXqSioeES/58vEmTJskwDMu+VatW6dJLL1VERIRCQ0PVpEkTPfjgg+7jp5ojuWbNGl122WUKCQlRRESE+vbtqx9//PGk99u5c6eGDBmiiIgIhYeHa+jQoTpy5Mip39gTXH/99froo4906NAh977Nmzdrx44duv7660u1P3jwoMaOHasWLVooNDRULpdLvXv31rfffutus3btWl100UWSpKFDh7qHyEteZ+fOnXXBBRdoy5Yt6tixo6pUqeJ+X06cI5mQkKCgoKBSr79nz56qWrWq9u7da/u1Ajg3UEgC0NKlS9WgQQP985//tNX+1ltv1cSJE9WmTRs9/fTT6tSpk6ZNm6bBgweXartz504NGjRIl19+uWbOnKmqVatqyJAh2rZtmyRpwIABevrppyVJ1113nV577TXNmjWrTP3ftm2brrjiCuXm5ioxMVEzZ87UVVddpS+++OK053366afq2bOn9u3bp0mTJmnMmDHasGGDLrnkEu3evbtU+2uuuUbZ2dmaNm2arrnmGi1YsECTJ0+23c8BAwbIMAwtWrTIve/NN99U06ZN1aZNm1Ltf/75Zy1ZskRXXHGFnnrqKY0bN05bt25Vp06d3EVds2bNlJiYKEm6/fbb9dprr+m1115Tx44d3dc5cOCAevfurdatW2vWrFnq0qXLSfv3zDPPqEaNGkpISFBhYaEk6cUXX9Qnn3yi5557TjExMbZfK4BzhAngnJaZmWlKMvv27WurfUpKiinJvPXWWy37x44da0oy16xZ494XGxtrSjLXrVvn3rdv3z7T6XSa9913n3vfL7/8Ykoyn3jiCcs1ExISzNjY2FJ9eOSRR8zj//P19NNPm5LM/fv3n7LfJfeYP3++e1/r1q3NmjVrmgcOHHDv+/bbb02Hw2HefPPNpe53yy23WK7Zv39/s1q1aqe85/GvIyQkxDRN0xw0aJDZrVs30zRNs7Cw0IyOjjYnT5580vfg2LFjZmFhYanX4XQ6zcTERPe+zZs3l3ptJTp16mRKMufOnXvSY506dbLsW7lypSnJfPTRR82ff/7ZDA0NNfv16/eXrxHAuYlEEjjHZWVlSZLCwsJstV+xYoUkacyYMZb99913nySVmkvZvHlzXXbZZe6fa9SooSZNmujnn3/2uM8nKplb+a9//UtFRUW2zklLS1NKSoqGDBmiyMhI9/6WLVvq8ssvd7/O4915552Wny+77DIdOHDA/R7acf3112vt2rVKT0/XmjVrlJ6eftJhbal4XqXDUfyf6cLCQh04cMA9bP/111/bvqfT6dTQoUNtte3Ro4fuuOMOJSYmasCAAQoKCtKLL75o+14Azi0UksA5zuVySZKys7Nttf/111/lcDjUqFEjy/7o6GhFRETo119/teyvW7duqWtUrVpVv//+u4c9Lu3aa6/VJZdcoltvvVVRUVEaPHiw3n333dMWlSX9bNKkSaljzZo102+//abDhw9b9p/4WqpWrSpJZXotffr0UVhYmN555x298cYbuuiii0q9lyWKior09NNPq3HjxnI6napevbpq1Kih7777TpmZmbbv+Y9//KNMD9Y8+eSTioyMVEpKip599lnVrFnT9rkAzi0UksA5zuVyKSYmRt9//32ZzjvxYZdT8fPzO+l+0zQ9vkfJ/L0SwcHBWrdunT799FPddNNN+u6773Tttdfq8ssvL9X2TJzJaynhdDo1YMAALVy4UIsXLz5lGilJU6dO1ZgxY9SxY0e9/vrrWrlypVatWqXzzz/fdvIqFb8/ZfHNN99o3759kqStW7eW6VwA5xYKSQC64oortGvXLm3cuPEv28bGxqqoqEg7duyw7M/IyNChQ4fcT2B7Q9WqVS1POJc4MfWUJIfDoW7duumpp57SDz/8oMcee0xr1qzRZ599dtJrl/Rz+/btpY799NNPql69ukJCQs7sBZzC9ddfr2+++UbZ2dknfUCpxPvvv68uXbooKSlJgwcPVo8ePdS9e/dS74ndot6Ow4cPa+jQoWrevLluv/12zZgxQ5s3b/ba9QH8vVBIAtD999+vkJAQ3XrrrcrIyCh1fNeuXXrmmWckFQ/NSir1ZPVTTz0lSYqPj/davxo2bKjMzEx999137n1paWlavHixpd3BgwdLnVuyMPeJSxKVqFWrllq3bq2FCxdaCrPvv/9en3zyift1locuXbpoypQpev755xUdHX3Kdn5+fqXSzvfee0//+9//LPtKCt6TFd1lNX78eKWmpmrhwoV66qmnVK9ePSUkJJzyfQRwbmNBcgBq2LCh3nzzTV177bVq1qyZ5ZttNmzYoPfee09DhgyRJLVq1UoJCQmaN2+eDh06pE6dOumrr77SwoUL1a9fv1MuLeOJwYMHa/z48erfv7/uvfdeHTlyRC+88ILOO+88y8MmiYmJWrduneLj4xUbG6t9+/Zpzpw5ql27ti699NJTXv+JJ55Q7969FRcXp2HDhuno0aN67rnnFB4erkmTJnntdZzI4XDooYce+st2V1xxhRITEzV06FD985//1Nat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va1fv959bPTo0Vq6dKnee+89JScna+/evRowYID7eGFhoeLj45WXl6cNGzZo4cKFWrBggSZOnFjmfpS5kFy3bp369OlTan/v3r21bt26MncAAADAV5TMkfT2JklZWVmWLTc395T98Pf3V3R0tHurXr26JCkzM1NJSUl66qmn1LVrV7Vt21bz58/Xhg0btGnTJknSJ598oh9++EGvv/66Wrdurd69e2vKlCmaPXu28vLK9u1BZS4kc3JyTrrMT0BAgLKyssp6OQAAAEiqU6eOwsPD3du0adNO2XbHjh2KiYlRgwYNdMMNNyg1NVWStGXLFuXn56t79+7utk2bNlXdunW1ceNGSdLGjRvVokULRUVFudv07NlTWVlZ2rZtW5n6XOZ1JFu0aKF33nmnVPz59ttvq3nz5mW9HAAAgM8wDO+v+1hyvT179sjlcrn3O53Ok7Zv3769FixYoCZNmigtLU2TJ0/WZZddpu+//17p6ekKDAxURESE5ZyoqCilp6dLktLT0y1FZMnxkmNlUeZC8uGHH9aAAQO0a9cude3aVZK0evVqvfnmm3r//ffLejkAAACf4TAMObxcSZZcz+VyWQrJU+ndu7f7zy1btlT79u0VGxurd999V8HBwV7t218p89D2lVdeqSVLlmjnzp266667dN999+l///uf1qxZo0aNGpVHHwEAAHAKEREROu+887Rz505FR0crLy9Phw4dsrTJyMhQdHS0JCk6OrrUU9wlP5e0savMhaQkxcfH64svvtDhw4f1888/65prrtHYsWPVqlUrTy4HAADgExzltJ2JnJwc7dq1S7Vq1VLbtm0VEBCg1atXu49v375dqampiouLkyTFxcVp69at2rdvn7vNqlWr5HK5yjxN0eO+r1u3TgkJCYqJidHMmTPVtWtX99NAAAAAKB9jx45VcnKydu/erQ0bNqh///7y8/PTddddp/DwcA0bNkxjxozRZ599pi1btmjo0KGKi4tThw4dJEk9evRQ8+bNddNNN+nbb7/VypUr9dBDD2nEiBGnnJd5KmWaI5menq4FCxYoKSlJWVlZuuaaa5Sbm6slS5bwoA0AAPjbK8+Hbez673//q+uuu04HDhxQjRo1dOmll2rTpk2qUaOGpOKvtXY4HBo4cKByc3PVs2dPzZkzx32+n5+fli1bpuHDhysuLk4hISFKSEhQYmJi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9PLLL8swDEnS/PnzFRERobVr16pHjx6aNWuWJkyYoAEDBkiS5s6dq5UrV57yvv/5z3/07rvvatWqVerevbskqUGDBu7jJcPgNWvWVEREhKTiBHPq1Kn69NNPFRcX5z5n/fr1evHFF9WpUye98MILatiwoWbOnClJatKkibZu3arHH3/ci+8aAJQvCkkAldqyZcsUGhqq/Px8FRUV6frrr9ekSZM0YsQItWjRwjIv8ttvv9XOnTsVFhZmucaxY8e0a9cuZWZmKi0tTe3bt3cf8/f3V7t27UoNb5dISUmRn5+fOnXqZLvPO3fu1JEjR3T55Zdb9ufl5enCCy+UJP3444+WfkhyF50A4CsoJAFUal26dNELL7ygwMBAxcTEyN//z/9shYSEWNrm5OSobdu2euONN0pdp0aNGh7dPzg4uMzn5OTkSJKWL1+uf/zjH5ZjTqfTo34AQGVEIQmgUgsJCVGjRo1stW3Tpo3eeecd1axZUy6X66RtatWqpS+//FIdO3aUJBUUFGjLli1q06bNSdu3aNFCRUVFSk5Odg9tH68kES0sLHTva968uZxOp1JTU0+ZZDZr1kwffvihZd+mTZv++kUCQCXCwzYA/jZuuOEGVa9eXX379tXnn3+uX375RWvXrtW9996r//73v5KkkSNHavr06VqyZIl++ukn3XXXXaddA7JevXpKSEjQLbfcoiVLlriv+e6770qSYmNjZRiGli1bpv379ysnJ0dhYWEaO3asRo8erYULF2rXrl36+uuv9dxzz2nhwoWSpDvvvFM7duzQuHHjtH37dr355ptasGBBeb9FAOBVFJIA/jaqVKmidevWqW7duhowYICaNWumYcOG6dixY+6E8r777tNNN92khIQExcXFKSwsTP379z/tdV944QUNGjRId911l5o2barbbrtNhw8fliT94x//0OTJk/XAAw8oKipKd999tyRpypQpevjhhzVt2jQ1a9ZMvXr10vLly1W/fn1JUt26dfXBBx9oyZIlatWqlebOnaupU6eW47sDAN5nmKeaYQ4AAACcBokkAAAAPEIhCQAAAI9QSAIAAMAjFJIAAADwCIUkAAAAPEIhCQAAAI9QSAIAAMAjFJIAAADwCIUkAAAAPEIhCQAAAI9QSAIAAMAj/w9QvXZlTzGe/AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "Report(\"XGBClassifier\", XGBClassifier( max_depth =500), **input)\n",
    "Report(\"XGBClassifier\", PassiveAggressiveClassifier(), **input)\n",
    "Report(\"XGBClassifier\", RidgeClassifier( ), **input)\n",
    "Report(\"XGBClassifier\", SGDClassifier(), **input)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################\n",
    "##########fill ablove\n",
    "#########################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Random Forests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* The accuracy on training set is **4% higher than the accuracy on test set which indicates a slight overfitting.** \n",
    "* We can decrease the depth of a tree in the forest because as trees get deeper, they tend to be more specific which results in not generalizing well.\n",
    "* However, reducing tree depth may also decrease the accuracy. \n",
    "* So we need to be careful when optimizing the parameters. \n",
    "* We can also increase the number of trees in the forest which will help the model to be more generalized and thus reduce overfitting. \n",
    "* **Parameter tuning is a very critical part in almost every project.**\n",
    "\n",
    "\n",
    "\n",
    "Another way is to do cross-validation which allows to use every sample in training and test set. \n",
    "\n",
    "GridSearchCV makes this process easy to handle. We can both do cross-validation and try different parameters using GridSearchCV."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **cv = 5 means having a 5-fold cross validation.**\n",
    "* So dataset is divided into 5 subset. \n",
    "* At each iteration, 4 subsets are used in training and the other subset is used as test set. \n",
    "* When 5 iteration completed, the model used all samples as both training and test samples.\n",
    "\n",
    "n_jobs parameter is used to select how many processors to use. -1 means using all processors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clf.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clf.best_score_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* We have achieved an overall accuracy of almost **90%.** \n",
    "* This is the mean cross-validated score of the best_estimator. \n",
    "* In the previous random forest, the mean score was approximately 86% (88% on training and 84% on test). \n",
    "* Using GridSearchCV, we improved the model accuracy by 4%."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
