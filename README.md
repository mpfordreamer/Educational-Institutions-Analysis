# **Data Science Project: Improving Student Retention and Academic Success at Jaya Jaya Institute Education**

## **Business Understanding**
![jayajaya](https://github.com/user-attachments/assets/9205e4a5-4db9-4f16-8a0a-b509f177d367)

**Jaya Jaya Institute Education** is currently facing a significant challenge related to its student dropout, enrolled, and graduate rates. A high dropout rate not only harms the institution's reputation but also wastes the resources invested in student education. With a deeper understanding of the factors influencing a student's decision to continue or drop out, the institution can design and implement smarter, more effective intervention strategies to improve retention and ensure higher academic achievement.

### **Key Business Problems:**

Despite the variety of study programs offered by Jaya Jaya Institute Education, there is a notable variation in retention rates and academic performance across different majors. The high dropout rates and low academic performance in some programs indicate that there are underlying factors that have not been fully understood or addressed. Therefore, it is necessary to develop a predictive classification model capable of anticipating a student's academic status based on their initial enrollment data and academic record. Based on this need, we have formulated the following core business questions:

1.  **What are the most significant factors influencing a student's academic status (Dropout, Enrolled, Graduate) at Jaya Jaya Institute Education?**
2.  **How accurately can the developed classification model predict a student's academic status?**
3.  **What strategic steps can the institution take to optimize retention and academic success based on insights from the predictive model?**

### **Project Scope:**

This project covers a comprehensive series of stages, from data analysis to model implementation:

*   In-depth analysis and exploration of the student dataset.
*   Rigorous data cleaning and preparation.
*   Building, training, and evaluating machine learning models with hyperparameter optimization using **Optuna**.
*   Interactive visualization of analysis results and model performance using **Looker Studio**.
*   Development of a real-time prediction application based on **Streamlit** to facilitate the model's use by the institution.

### **Data Preparation**

#### **Dataset:**

The dataset that forms the foundation of this project is sourced from a GitHub repository, accessible via the following link:
**Dataset Link:** [https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Dataset Column Details:**
-   **Marital status**: The marital status of the student.
-   **Application mode**: The method of application used by the student.
-   **Application order**: The order in which the student applied.
-   **Course**: The course taken by the student.
-   **Daytime/evening attendance**: Whether the student attends classes during the day or in the evening.
-   **Previous qualification**: The qualification obtained by the student before enrolling in higher education.
-   **Previous qualification (grade)**: Grade of previous qualification (between 0 and 200)
-   **Nacionality**: The nationality of the student.
-   **Mother's qualification**: The qualification of the student's mother.
-   **Father's qualification**: The qualification of the student's father.
-   **Mother's occupation**: The occupation of the student's mother.
-   **Father's occupation**: The occupation of the student's father.
-   **Admission grade**: Admission grade (between 0 and 200)
-   **Displaced**: Whether the student is a displaced person.
-   **Educational special needs**: Whether the student has any special educational needs.
-   **Debtor**: Whether the student is a debtor.
-   **Gender**: The gender of the student.
-   **Scholarship holder**: Whether the student is a scholarship holder.
-   **Age at enrollment**: The age of the student at the time of enrollment.
-   **International**: Whether the student is an international student.
-   **Curricular units 1st sem (credited)**: The number of curricular units credited by the student in the first semester.
-   **Curricular units 1st sem (enrolled)**: The number of curricular units enrolled by the student in the first semester.
-   **Curricular units 1st sem (evaluations)**: The number of curricular units evaluated by the student in the first semester.
-   **Curricular units 1st sem (approved)**: The number of curricular units approved by the student in the first semester.

Ensure your environment matches `requirements.txt` before performing data preparation:
```bash
pip install -r requirements.txt
```

#### **Data Preparation Process:**
The data preparation stages were carried out systematically:
1.  **Anomaly Detection**: Performing an initial check to identify anomalies or outliers in the dataset.
2.  **Target Distribution Analysis**: Examining the distribution of the target column (`Dropout`, `Enrolled`, `Graduate`) to determine the need for data balancing (it was found to be imbalanced, requiring balancing).
3.  **Target Column Encoding**: Converting the target column into a numerical representation using `LabelEncoder`.
4.  **Feature Correlation Analysis**: Creating a correlation heatmap to visualize the relationships between features and the target column.
5.  **Feature Selection**: Selecting a subset of features that show a significant correlation with the target column for more efficient modeling.
6.  **Data Splitting & Scaling**: Splitting the dataset into training (80%) and testing (20%) sets, followed by scaling the features to ensure they are on a uniform scale.
7.  **Balancing the Target Data**: Applying a data balancing technique to the imbalanced target column to prevent model bias.

### **Modeling**

In the effort to build a robust predictive model, three ensemble-based machine learning algorithms were trained and optimized using **Optuna** to predict the academic status of students (Dropout, Enrolled, Graduate):

*   **ExtraTreesClassifier**: An ensemble method that builds multiple decision trees with additional randomness in both feature selection and data splitting.
*   **Random Forest Classifier**: Another ensemble method that improves prediction accuracy and reduces overfitting by aggregating multiple decision trees.
*   **XGBoost (Extreme Gradient Boosting)**: A highly efficient and high-performing boosting algorithm, known for its ability to deliver accurate predictions at high speed.

Based on a comparative evaluation, the **XGBoostClassifier** model showed the most competitive and superior performance among the trained models. This best model will then be used to make predictions on future student data.

### **Evaluation**

The evaluation metrics used to measure predictive performance include:

*   **Accuracy**: Measures the proportion of total correct predictions.
*   **Precision & Recall**: Analyzes the trade-off between accurate positive identifications (Precision) and the coverage of actual positives (Recall).
*   **F1-Score**: Provides a balance between Precision and Recall, which is highly relevant for classification problems with imbalanced classes.
*   **Confusion Matrix**: A detailed visual representation of the model's performance in classifying each class, helping to identify areas of misclassification.

Here is a summary of the performance evaluation results for the optimized models:

| Model                                 | Accuracy Testing | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|---------------------------------------|------------------|-----------------|--------------|----------------|
| ExtraTrees (Optimized)                | 75.14%           | 0.73            | 0.69         | 0.69           |
| RandomForest (Optimized)              | 75.71%           | 0.71            | 0.69         | 0.70           |
| **XGBoost (Optimized)**               | **77.00%**       | **0.73**        | **0.69**     | **0.70**       |

**Detailed Performance of the Best Model (XGBoostClassifier):**

![image](https://github.com/user-attachments/assets/1172b612-65e6-4a2c-b3bf-cb0f56961682)

The XGBoost model, optimized with the best hyperparameters (`n_estimators`: 381, `learning_rate`: 0.077, `max_depth`: 10, `min_child_weight`: 1, `subsample`: 0.664, `colsample_bytree`: 0.885), achieved a **test accuracy of 0.77 (or 77%)**. The confusion matrix and classification report show the following performance in classifying student status:

*   **Graduate (Class 1):** Predicted **very well**. The model identified **409 out of 442** Graduate students (recall of **0.88**), resulting in the highest F1-score (**0.85**). Misclassifications were minimal (**6** as Dropout, **27** as Enrolled).
*   **Dropout (Class 0):** Predicted **well**. The model identified **209 out of 284** Dropout students (recall of **0.70**), with an F1-score of **0.76**. The main errors were misclassifying them as Enrolled (**38**) and Graduate (**37**).
*   **Enrolled (Class 2):** The **most challenging** class to predict. The model identified **65 out of 159** Enrolled students (recall of **0.48**), resulting in an F1-score of **0.46**. It was often misclassified as Graduate (**61**) and Dropout (**33**).

The overall model accuracy is **0.77**. The model excels at identifying 'Graduate', is quite good for 'Dropout', and shows weakness in predicting 'Enrolled'.

---

## **Business Dashboard**

To provide interactive and easily accessible business insights, a business dashboard has been created using **Looker Studio**. This dashboard allows stakeholders to dynamically explore the data and analysis results.

**Access the Interactive Dashboard (Looker Studio):** [https://lookerstudio.google.com/reporting/8021730a-9141-49b4-8397-c12e86d1e78b](https://lookerstudio.google.com/reporting/8021730a-9141-49b4-8397-c12e86d1e78b)

## **Running the Machine Learning System**

The machine learning system can be run via a Python script or accessed directly through an interactive web application:

1.  **Prepare the Data**: Ensure the required dataset (e.g., `data_student.csv`) is available in the appropriate folder, or prepare new data you wish to predict.
2.  **Run the Model from a Python Script**: To make predictions with the trained **XGBoost** model, use the following command in the terminal:
    ```bash
    python app.py --model xgboost_model.pkl --input data_student.csv
    ```
3.  **Access the Web Application (Streamlit)**: For a more user-friendly, real-time experience, you can access the prediction application directly via the following link:
    **Prediction App (Streamlit):** [https://jayajaya-educational-institutions-analysis-mpfordreamer.streamlit.app/](https://jayajaya-educational-institutions-analysis-mpfordreamer.streamlit.app/)

## **Conclusion**

This project successfully achieved its main goal: to build a robust classification model to predict the academic status of students (Dropout, Graduate, Enrolled) based on their demographic and early performance data.

---

**Key Factors Determining Academic Status**
Based on the feature importance analysis from the **XGBoost** model (see the "Feature Importance from XGBoost Model" chart), several key factors that most significantly influence a student's academic status are:
*   `Tuition_fees_up_to_date`: As the most dominant predictor, this highlights the importance of timely tuition payments.
*   `Curricular_units_2nd_sem_approved`: The number of credits approved in the second semester is the second most crucial contributor.
*   `Scholarship_holder`: The status as a scholarship recipient also plays a significant role.
*   Other features like `Curricular_units_1st_sem_approved`, `Age_at_enrollment`, and `Debtor` show equally important contributions.

---

**Best Predictive Model (XGBoostClassifier)**
After careful hyperparameter optimization, the selected **XGBoostClassifier** model showed strong performance on the test data, with the following details:
*   **Overall Accuracy**: The model achieved an accuracy of **77.00%**, making it the best-performing model in this project.
*   **Performance by Class:**
    *   **Graduate**: Predicted **very well** (Recall: **0.88**, F1-Score: **0.85**). The model successfully identified **409 out of 442** students who actually graduated.
    *   **Dropout**: Predicted **fairly well** (Recall: **0.70**, F1-Score: **0.76**). The model successfully identified **209 out of 284** students who dropped out.
    *   **Enrolled**: Remains the **most challenging** class to predict (Recall: **0.48**, F1-Score: **0.46**). The model successfully identified **65 out of 159** students who were still enrolled.

The model proved to be very effective at predicting graduation and reasonably reliable at detecting dropout risk. However, the accuracy for predicting students who are still 'Enrolled' shows room for future improvement.

Overall, this project successfully answered the business questions and met most of the established objectives. The **XGBoostClassifier** model, with its strong predictive performance and feature interpretability, is an ideal candidate for implementation in the deployment phase, with potential for further performance enhancements.

### **Strategic Action Recommendations for the Educational Institution**

Based on the conclusions and insights from the predictive model, here are recommended actions that Jaya Jaya Institute Education can implement to significantly reduce the dropout rate and improve student academic success:

1.  **Proactive Financial Support**:
    *   Implement an early warning system to identify students who may have difficulty paying tuition fees.
    *   Offer flexible payment options, installment plans, or emergency financial aid to prevent financial issues from leading to dropout.

2.  **Academic Monitoring in Early Semesters**:
    *   Closely monitor students' academic performance, especially in first and second-semester courses.
    *   Provide targeted tutoring programs, study sessions, or academic counseling as soon as a decline in performance is detected.

3.  **Scholarship Program Expansion and Promotion**:
    *   Increase the scope and effectiveness of scholarship programs and promote them widely to prospective students.
    *   Scholarships not only alleviate financial burdens but can also boost students' motivation and commitment to their studies.

4.  **Realistic Workload Management**:
    *   Encourage more intensive academic counseling to help students plan a realistic number of credits based on their abilities and readiness.
    *   Provide extra support for students who are struggling to complete the courses they have enrolled in.

5.  **Holistic Mentoring and Counseling Programs**:
    *   Provide comprehensive personal mentoring and counseling programs that consider diverse student backgrounds (e.g., age at enrollment, nationality).
    *   This approach is crucial for addressing adaptational, social, or psychological challenges that at-risk students may face, preventing them from leading to dropout.

---
