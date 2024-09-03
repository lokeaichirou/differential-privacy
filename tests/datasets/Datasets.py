
import numpy as np
import pandas as pd
from sklearn import datasets


class Datasets():

    def __init__(self):
        pass

    @staticmethod
    def standardize_attribute_without_rescaling_by_sigma(data_column):
        return data_column - data_column.mean()

    @staticmethod
    def standardize_attribute(data_column):
        return (data_column - data_column.mean())/data_column.std()

    @staticmethod
    def process_data_as_in_functional_mech_paper(data, transform_columns, process_columns):

        max_frobenius_norm      = np.linalg.norm(data[transform_columns], axis=1).max()
        data[transform_columns] = data[transform_columns].transform(lambda x: x / max_frobenius_norm)

        for col in process_columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min()) * 2 - 1

        return data


class Insurance_dataset(Datasets):

    def __init__(self, csv_file_address):
        super().__init__()
        self.original_data = pd.read_csv(csv_file_address)

    def pre_processing_data(self):
        categorical_columns = ['sex', 'children', 'smoker', 'region']

        # pre-processing
        df_encode           = pd.get_dummies(data       = self.original_data,
                                             prefix     = 'OHE',
                                             prefix_sep = '_',
                                             columns    = categorical_columns,
                                             drop_first = True,
                                             dtype      = 'int8')

        df_encode['charges'] = np.log(df_encode['charges'])

        return df_encode

    def get_pre_processed_data(self, further_processing_method = None):
        pre_processed_data = self.pre_processing_data()

        if further_processing_method == 'standardize':
            pre_processed_data['age'] = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['age'])
            pre_processed_data['bmi'] = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['bmi'])

        elif further_processing_method == 'functional_mech_paper':
            pre_processed_data = self.process_data_as_in_functional_mech_paper(pre_processed_data,
                                                                               ['age', 'bmi', 'OHE_male', 'OHE_1', 'OHE_2', 'OHE_3', 'OHE_4', 'OHE_5', 'OHE_yes', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest'],
                                                                               ['charges'])

        return pre_processed_data.drop('charges', axis=1), pre_processed_data['charges']


class Housing_dataset(Datasets):

    def __init__(self, csv_file_address):
        super().__init__()
        self.original_data = pd.read_csv(csv_file_address)

    def pre_processing_data(self):
        pre_processed_data = self.original_data

        bedroom_median     = pre_processed_data['total_bedrooms'].median()
        pre_processed_data['total_bedrooms'].fillna(bedroom_median, inplace=True)

        # For total_bedrooms
        IQR                = pre_processed_data['total_bedrooms'].quantile(0.75) - pre_processed_data['total_bedrooms'].quantile(0.25)
        newB               = pre_processed_data['total_bedrooms'].quantile(0.75) + 3 * (IQR)
        pre_processed_data.drop(pre_processed_data[pre_processed_data['total_bedrooms'] > newB].index, axis=0, inplace=True)

        # For population
        IQR_p              = pre_processed_data['population'].quantile(0.75) - pre_processed_data['population'].quantile(0.25)
        newB_p             = pre_processed_data['population'].quantile(0.75) + 3 * (IQR_p)
        pre_processed_data.drop(pre_processed_data[pre_processed_data['population'] > newB_p].index, axis=0, inplace=True)

        # For household
        IQR_h              = pre_processed_data['households'].quantile(0.75) - pre_processed_data['households'].quantile(0.25)
        newB_h             = pre_processed_data['households'].quantile(0.75) + 3 * (IQR_h)
        pre_processed_data.drop(pre_processed_data[self.original_data['households'] > newB_h].index, axis=0, inplace=True)

        # For total_rooms
        IQR_t              = pre_processed_data['total_rooms'].quantile(0.75) - pre_processed_data['total_rooms'].quantile(0.25)
        newB_t             = pre_processed_data['total_rooms'].quantile(0.75) + 3 * (IQR_t)
        pre_processed_data.drop(pre_processed_data[pre_processed_data['total_rooms'] > newB_t].index, axis=0, inplace=True)


        pre_processed_data = pd.get_dummies(pre_processed_data, prefix=None)

        return pre_processed_data

    def get_pre_processed_data(self, further_processing_method = None):
        pre_processed_data = self.pre_processing_data()

        if further_processing_method == 'standardize':
            pre_processed_data['housing_median_age'] = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['housing_median_age'])
            pre_processed_data['total_rooms']        = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['total_rooms'])
            pre_processed_data['total_bedrooms']     = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['total_bedrooms'])
            pre_processed_data['population']         = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['population'])
            pre_processed_data['households']         = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['households'])
            pre_processed_data['median_income']      = self.standardize_attribute_without_rescaling_by_sigma(pre_processed_data['median_income'])

        elif further_processing_method == 'functional_mech_paper':
            pre_processed_data = self.process_data_as_in_functional_mech_paper(pre_processed_data,
                                                                               ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'],
                                                                               ['median_house_value'])

        return pre_processed_data.drop(columns = ["median_house_value", "longitude", "latitude", "ocean_proximity_ISLAND"]).astype(float), pre_processed_data.loc[:, 'median_house_value'].astype(float)


class HandWrittenDigits_dataset(Datasets):

    def __init__(self):
        super().__init__()

    def get_raw_data(self, mnist=False):

        if mnist:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y  = mnist["data"], mnist["target"]
            y     = y.astype(np.uint8)
        else:
            digits_df = datasets.load_digits()
            X         = digits_df.data
            y         = digits_df.target

        return X, y


class Titanic_dataset(Datasets):

    def __init__(self, csv_file_address):
        super().__init__()
        self.original_data = pd.read_csv(csv_file_address)

    def pre_processing_data(self):
        categorical_columns = ['sex', 'children', 'smoker', 'region']

        # Generate Title feature based on Name feature
        train_df          = self.original_data.drop(['Ticket', 'Cabin'], axis=1)
        train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
        train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
        train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

        title_mapping     = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        train_df['Title'] = train_df['Title'].map(title_mapping)
        train_df['Title'] = train_df['Title'].fillna(0)

        # Drop useless Name and PassengerId features
        train_df          = train_df.drop(['Name', 'PassengerId'], axis=1)

        # Convert categorical Sex feature into categorical values
        train_df['Sex']   = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        # Fill missing age values with guessed value(median value)
        guess_ages = np.zeros((2,3))

        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = train_df[(train_df['Sex'] == i) & (train_df['Pclass'] == j+1)]['Age'].dropna()

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                train_df.loc[ (train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j+1), 'Age'] = guess_ages[i,j]

        train_df['Age'] = train_df['Age'].astype(int)

        # Convert numerical age feature into categorical values
        train_df.loc[ train_df['Age'] <= 16, 'Age'] = 0
        train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
        train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
        train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
        train_df.loc[ train_df['Age'] > 64, 'Age']                            = 4

        # Create IsAlone feature based on SibSp and Parch features
        train_df['FamilySize']                               = train_df['SibSp'] + train_df['Parch'] + 1
        train_df['IsAlone']                                  = 0
        train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

        train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

        # Generate Age*Class feature
        train_df['Age*Class'] = train_df.Age * train_df.Pclass

        # Fill up the missed Embarked feature values with most frequent value and then convert into categorical values
        train_df['Embarked'] = train_df['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])
        train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        #
        train_df.loc[ train_df['Fare'] <= 7.91, 'Fare']                                = 0
        train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
        train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
        train_df.loc[ train_df['Fare'] > 31, 'Fare']                                   = 3
        train_df['Fare']                                                               = train_df['Fare'].astype(int)

        return train_df


    def get_pre_processed_data(self, further_processing_method = None):
        pre_processed_data = self.pre_processing_data()

        return pre_processed_data.drop("Survived", axis=1), pre_processed_data["Survived"]


if __name__ == '__main__':
    insurance_data = Insurance_dataset("insurance.csv")
    insurance_data.get_pre_processed_data(further_processing_method='standardize')
