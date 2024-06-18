import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title of the web app
st.title('🩺 Cancer Prediction Dataset ')
st.write('Submitted by: Rosalie Tagud and Dhanica Shelly Ganibe')

# Adding some text to the app
st.write('The nature of this data is Predicting Cancer Risk from Medical and Lifestyle Data',
          'and Disclaimer this dataset has been preprocessed and cleaned to ensure that users can focus on the most critical aspects of their analysis.', 
          'The preprocessing steps were designed to eliminate noise and irrelevant information, allowing you to concentrate on developing ',
          'and fine-tuning your predictive models.'
          )

# Load dataset
df = pd.read_csv("The_Cancer_data_1500_V2.csv")
st.dataframe(df)

# Print the column names to verify them
st.write("Column Names:", df.columns.tolist())

def main():
    st.sidebar.title("Data Visualization")

    pages = {
        "Home": home,
        "Number of People With Cancer vs No Cancer": number_of_people_with_cancer,
        "Positive Diagnoses Ratio": positive_diagnoses_ratio,
        "Number of Male and Female": number_of_male_and_female,
        "BMI of People With Cancer": bmi_of_people_with_cancer,
        "Number of People Smoking": number_of_people_smoking,
        "Cancer History": cancer_history,
        "Classifier Plot": classifier_plot,
    }

    selection = st.sidebar.selectbox("Select Data", list(pages.keys()))

    # Display the selected page
    page = pages[selection]
    page()


def home():
    st.header("Home")
    st.write("About Data")
    st.write(
        """
        This dataset contains medical and lifestyle information for 1500 patients, 
        designed to predict the presence of cancer based on various features. 
        The dataset is structured to provide a realistic challenge for predictive modeling in the medical domain.

    
        **Dataset Structure**
        - Age: Integer values representing the patient's age, ranging from 20 to 80.
        - Gender: Binary values representing gender, where 0 indicates Male and 1 indicates Female.
        - BMI: Continuous values representing Body Mass Index, ranging from 15 to 40.
        - Smoking: Binary values indicating smoking status, where 0 means No and 1 means Yes.
        - GeneticRisk: Categorical values representing genetic risk levels for cancer, with 0 indicating Low, 1 indicating Medium, and 2 indicating High.
        - PhysicalActivity: Continuous values representing the number of hours per week spent on physical activities, ranging from 0 to 10.
        - AlcoholIntake: Continuous values representing the number of alcohol units consumed per week, ranging from 0 to 5.
        - CancerHistory: Binary values indicating whether the patient has a personal history of cancer, where 0 means No and 1 means Yes.
        - Diagnosis: Binary values indicating the cancer diagnosis status, where 0 indicates No Cancer and 1 indicates Cancer.
        """
    )

def number_of_people_with_cancer():
    st.header("Number of People With Cancer vs No Cancer")
    
    # Count the number of people with and without cancer
    cancer_counts = df['Diagnosis'].value_counts()
    
    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df[df['Diagnosis'] == 0]['Diagnosis'], bins=2, color='skyblue', label='No Cancer')
    plt.hist(df[df['Diagnosis'] == 1]['Diagnosis'], bins=2, color='salmon', label='Cancer')
    plt.xlabel('Diagnosis (0: No Cancer, 1: Cancer)', fontsize=12)
    plt.ylabel('Number of People', fontsize=12)
    plt.title('Number of People With and Without Cancer', fontsize=14)
    plt.legend()
    st.pyplot()


    st.write('It looks like smoking has impact to cancer risk:',
             'from P(Cancer|No smoking) = 30.5 % up to P(Cancer|Smoking) = 55.2 %',
             'Be carefull smoking increases the risk of cancer by 25 %!'
           )
   
def positive_diagnoses_ratio():
    st.header("Positive Diagnoses Ratio by Gender")
    
    # Calculate positive diagnoses ratio by gender
    diagnoses_by_gender = df.groupby('Gender')['Diagnosis'].mean()
    
    # Plotting the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=diagnoses_by_gender.index, y=diagnoses_by_gender.values)
    plt.xlabel('Gender (0: Male, 1: Female)', fontsize=12)
    plt.ylabel('Positive Diagnoses Ratio', fontsize=12)
    plt.title('Positive Diagnoses Ratio by Gender', fontsize=14)
    st.pyplot()

    
    st.write('We have uniform gender distribution.',
             'But women are 24.2 % more likely to develop cancer:',
             'P(Cancer|Female) - P(Cancer|Male) = 24.2%'
           )
    

def number_of_male_and_female():
    st.header("Number of Male and Female")
    
    # Counting number of males and females
    gender_counts = df['Gender'].value_counts()
    
    # Plotting the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.xlabel('Gender (0: Male, 1: Female)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Number of Male and Female', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
    st.pyplot()

def bmi_of_people_with_cancer():
    st.header("BMI of People With Cancer")
    
    # Filter the dataframe for people with cancer
    cancer_df = df[df['Diagnosis'] == 1]
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(cancer_df['BMI'], bins=20, kde=True, color='skyblue')
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('BMI Distribution of People With Cancer', fontsize=14)
    st.pyplot()

    st.write('An increase in BMI above 25 significantly raises the risk of cancer.',
                'P(Cancer|BMI > 26) > 43 %',
                'P(Cancer|BMI ≤ 26) < 25 %'
           )


def number_of_people_smoking():
    st.header("Number of People Smoking")
    
    # Counting number of smokers and non-smokers
    smoking_counts = df['Smoking'].value_counts()
    
    # Plotting the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=smoking_counts.index, y=smoking_counts.values)
    plt.xlabel('Smoking Status (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Number of People Smoking', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['No Smoking', 'Smoking'])
    st.pyplot()


def cancer_history():
    st.header("Cancer History")
    
    # Counting number of people with and without cancer history
    history_counts = df['CancerHistory'].value_counts()
    
    # Plotting the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=history_counts.index, y=history_counts.values)
    plt.xlabel('Cancer History (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Cancer History of People', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['No Cancer History', 'Cancer History'])
    st.pyplot()

    
def classifier_plot():
    st.header("Classifier Output Plot")
    
    # Check if 'ClassifierScore' exists in df.columns
    if 'ClassifierScore' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df['ClassifierScore'], shade=True, color='skyblue', label='Classifier Output')
        plt.xlabel('Classifier Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Classifier Output', fontsize=14)
        plt.legend()
        st.pyplot()
    else:
        st.write("Column 'ClassifierScore' not found in the dataset.")

if __name__ == "__main__":
    main()

