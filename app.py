import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import streamlit as st
import altair as alt
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.data.path.append("/path/to/nltk_data")

from nltk.tokenize import sent_tokenize

# Load the sentence tokenizer once
nltk.download('punkt')



class DataMaker:
    def __init__(self, site, first_page, last_page):
        self.site_url = site
        self.first_page = first_page
        self.last_page = last_page
        self.urls = []
        self.courses = []
        self.organizations = []
        self.learning_products = []
        self.ratings = []
        self.num_rated = []
        self.difficulty = []
        self.enrolled = []
        self.skills = []  # New attribute to store skills

    def scrape_features(self, page_url):
        course_list_page = requests.get(page_url)
        course_list_soup = BeautifulSoup(course_list_page.content, 'html.parser')

        cnames = course_list_soup.select(".headline-1-text")
        for i in range(10):
            self.courses.append(cnames[i].text)

        pnames = course_list_soup.select(".horizontal-box > .partner-name")
        for i in range(10):
            self.organizations.append(pnames[i].text)

        root = "https://www.coursera.org"
        links = course_list_soup.select(".ais-InfiniteHits > .ais-InfiniteHits-list > .ais-InfiniteHits-item")
        for i in range(10):
            self.urls.append(root + links[i].a["href"])

        for i in range(10):
            learn_pdcts = course_list_soup.find_all('div', '_jen3vs _1d8rgfy3')
            self.learning_products.append(learn_pdcts[i].text)

        ratings = []
        num_ratings = []
        cratings = course_list_soup.select(".ratings-text")
        cnumratings = course_list_soup.select(".ratings-count")
        for i in range(10):
            try:
                self.ratings.append(float(cratings[i].text))
            except:
                self.ratings.append("Missing")
            try:
                self.num_rated.append(int(cnumratings[i].text.replace(',','').replace('(','').replace(')','')))
            except:
                self.num_rated.append("Missing")

        enrollers = course_list_soup.select(".enrollment-number")
        for i in range(10):
            try:
                self.enrolled.append(enrollers[i].text)
            except:
                self.enrolled.append("Missing")

        difficulty = course_list_soup.select(".difficulty")
        for i in range(10):
            self.difficulty.append(difficulty[i].text)
        
        # Scrape and store skills data
        cskills = course_list_soup.find_all("span", "_x4x75x")
        for idx in range(10):
            temp = ",".join([cskills[idx].text for skill in cskills])
            self.skills.append(temp)

    def crawler(self):
        for page in range(self.first_page, self.last_page+1):
            st.write("\nCrawling Page " + str(page))
            page_url = self.site_url + "?page=" + str(page) + "&index=prod_all_products_term_optimization"
            self.scrape_features(page_url)

    def make_dataset(self):
        self.crawler()
        data_dict = {
            "Course URL":self.urls,
            "Course Name":self.courses,
            "Learning Product Type":self.learning_products,
            "Course Provided By":self.organizations,
            "Course Rating":self.ratings,
            "Course Rated By":self.num_rated,
            "Enrolled Student Count":self.enrolled,
            "Course Difficulty":self.difficulty,
            "Skills": self.skills  # Add skills to the data dictionary
        }
        data = pd.DataFrame(data_dict)
        return data

class DataHunter:
    def __init__(self, df):
        self.df = df
        self.skills = []
        self.about = []
        self.new_career_starts = []
        self.pay_increase_prom = []
        self.estimate_toc = []
        self.instructors = []

    def scrape_features(self, page_url):
        course_page = requests.get(page_url)
        course_soup = BeautifulSoup(course_page.content, 'html.parser')
        
        # Scrape and store skills data
        cskills = course_soup.find_all("span", "_x4x75x")
        temp = ",".join([cskills[idx].text for skill in cskills])
        self.skills.append(temp)

        try:
            cdescr = course_soup.select(".description")
            self.about.append(cdescr[0].text)
        except:
            self.about.append("Missing")

        try:
            learn_stats = course_soup.select("._1qfi0x77 > .LearnerOutcomes__text-wrapper > .LearnerOutcomes__percent")
        except:
            pass
        try:
            self.new_career_starts.append((float(learn_stats[0].text.replace('%',''))))
        except:
            self.new_career_starts.append("Missing")
        try:
            self.pay_increase_prom.append((float(learn_stats[1].text.replace('%',''))))
        except:
            self.pay_increase_prom.append("Missing")

        try:
            props = course_soup.select("._16ni8zai")
            done = 0 
            etoc = "Missing"
            for idx in range(len(props)):
                if('to complete' in props[idx].text and done==0):
                    etoc = props[idx].text
                    done+=1
            self.estimate_toc.append(etoc)
        except:
            self.estimate_toc.append("Missing")

        try:
            instructors = course_soup.select(".instructor-name")
            temp=""
            for idx in range(len(instructors)):
                temp = temp + instructors[idx].text
                if(idx != len(instructors)-1):
                    temp = temp + ","
            self.instructors.append(temp)
        except:
            self.instructors.append("Missing")

    def extract_url(self):
        for url in self.df['Course URL']:
            self.scrape_features(url)

    def make_dataset(self):
        self.extract_url()
        data_dict = {
            "Skills":self.skills,
            "Description":self.about,
            "Percentage of new career starts":self.new_career_starts,
            "Percentage of pay increase or promotion":self.pay_increase_prom,
            "Estimated Time to Complete":self.estimate_toc,
            "Instructors":self.instructors
        }
        data = pd.DataFrame(data_dict)
        return data


@st.cache_data()
def load_data():
    source_path1 = os.path.join("coursera-courses-overview.csv")
    source_path2 = os.path.join("coursera-individual-courses.csv")
    df_overview = pd.read_csv(source_path1)
    df_individual = pd.read_csv(source_path2)
    df = pd.concat([df_overview, df_individual], axis=1)
    return df

def content_based_recommendations(df, input_course, courses):
    # Filter DataFrame to include only selected courses
    selected_courses_df = df[df['Course Name'].isin(courses)].reset_index(drop=True)
    
    # Extract keywords from course descriptions
    selected_courses_df['descr_keywords'] = extract_keywords(selected_courses_df, 'Description')

    # Check if any course descriptions are empty or contain only stop words
    non_empty_indices = selected_courses_df[selected_courses_df['descr_keywords'].apply(lambda x: len(x) > 0)].index

    if len(non_empty_indices) == 0:
        st.write("No valid course descriptions found for vectorization.")
        return

    # Vectorize the non-empty course descriptions
    count = CountVectorizer()
    count_matrix = count.fit_transform(selected_courses_df['descr_keywords'][non_empty_indices])

    if count_matrix.shape[0] == 0:
        st.write("No valid documents found for vectorization.")
        return

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Get recommendations
    rec_courses_similar = recommendations(selected_courses_df, input_course, cosine_sim, True)
    rec_courses_dissimilar = recommendations(selected_courses_df, input_course, cosine_sim, False)

    st.write("Top 5 most similar courses")
    st.write(selected_courses_df[selected_courses_df['Course Name'].isin(rec_courses_similar)])
    st.write("Top 5 most dissimilar courses")
    st.write(selected_courses_df[selected_courses_df['Course Name'].isin(rec_courses_dissimilar)])

    df = df[df['Course Name'].isin(courses)].reset_index()

    # Extract keywords from course descriptions
    df['descr_keywords'] = extract_keywords(df, 'Description')
    
    # Check if any course descriptions are empty or contain only stop words
    non_empty_indices = df[df['descr_keywords'].apply(lambda x: len(x) > 0)].index
    
    if len(non_empty_indices) == 0:
        st.write("No valid course descriptions found for vectorization.")
        return
    
    # Vectorize the non-empty course descriptions
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['descr_keywords'][non_empty_indices])
    
    if count_matrix.shape[0] == 0:
        st.write("No valid documents found for vectorization.")
        return
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Get recommendations
    rec_courses_similar = recommendations(df, input_course, cosine_sim, True)
    temp_sim = df[df['Course Name'].isin(rec_courses_similar)]
    rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, False)
    temp_dissim = df[df['Course Name'].isin(rec_courses_dissimilar)]

    st.write("Top 5 most similar courses")
    st.write(temp_sim)
    st.write("Top 5 most dissimilar courses")
    st.write(temp_dissim)


def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
    recommended = []
    selected_course = df[df['Course Name'] == input_course]
    idx = selected_course.index[0]

    if find_similar:
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    else:
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=True)

    if len(score_series) < how_many:
        how_many = len(score_series)
    top_sugg = list(score_series.iloc[1:how_many + 1].index)

    for i in top_sugg:
        qualified = df['Course Name'].iloc[i]
        recommended.append(qualified)

    return recommended

def extract_keywords(df, feature):
    r = Rake()
    keyword_lists = []
    for i in range(df[feature].shape[0]):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)

    return keyword_lists

def prep_for_cbr(df):
    st.header("Content-based Recommendation")
    st.sidebar.header("Filter on Preferences")
    st.write("This section is entrusted with the responsibility of"
             " analysing a filtered subset of courses based on the **skills**"
             " a learner is looking to develop. This filter can be adjusted on"
             " the sidebar.")
    st.write("This section also finds courses similar to a selected course"
             " based on Content-based recommendation. The learner can choose"
             " any course that has been filtered on the basis of their skills"
             " in the previous section.")
    st.write("Enter skills in the input box below separated by commas.")

    skills_avail = set()
    for i in range(len(df)):
        try:
            skills_avail.update(df['Skills'][i])
        except TypeError:
            pass

    skills_input = st.sidebar.text_input("Enter Skills", "", placeholder="e.g., Python, Data Analysis", key='skills_input')
    if skills_input:
        skills_select = [s.strip() for s in skills_input.split(",")]
    else:
        skills_select = []

    skill_filtered = filter(df, skills_select, 'Skills', 'Course URL')
    skill_filtered = df[df['Course URL'].isin(skill_filtered)].reset_index()

    courses = skill_filtered['Course Name']
    st.write("### Filtered courses based on skill preferences")
    st.write(skill_filtered)
    st.write("**Number of programmes filtered:**", len(skill_filtered))
    st.write("**Number of courses:**",
             len(skill_filtered[skill_filtered['Learning Product Type'] == 'COURSE']))
    st.write("**Number of professional degrees:**",
             len(skill_filtered[skill_filtered['Learning Product Type'] == 'PROFESSIONAL CERTIFICATE']))
    st.write("**Number of specializations:**",
             len(skill_filtered[skill_filtered['Learning Product Type'] == 'SPECIALIZATION']))
    chart = alt.Chart(skill_filtered).mark_bar().encode(
        y='Course Provided By:N',
        x='count(Course Provided By):Q'
    ).properties(
        title='Organizations providing these courses'
    )
    st.altair_chart(chart)

    if len(courses) <= 2:
        st.write("*There should be at least 3 courses. Do add more.*")

    input_course = st.sidebar.selectbox("Select Course", list(courses), key='courses')

    if input_course == "":
        st.write("Please select a course from the dropdown")
    else:
        rec_radio = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)
        if rec_radio == 'yes':
            content_based_recommendations(df, input_course, courses)



def filter(dataframe, chosen_options, feature, id):
    selected_records = []
    for i in range(len(dataframe)):
        # Check if the value is iterable (i.e., not float)
        if isinstance(dataframe[feature][i], str):
            for op in chosen_options:
                if op in dataframe[feature][i]:
                    selected_records.append(dataframe[id][i])
    return selected_records


def main():
    st.title("Coursera Courses Explorer")
    st.sidebar.title("Navigation")
    option = st.sidebar.radio('Go to', ['Home', 'About','Dataset Overview', 'Content-based Recommendation'])

    if option == 'Home':
        st.header("Welcome to Coursera Courses Explorer")
        st.write("This app allows you to explore courses on Coursera.")
        st.write("You can filter courses based on your preferences and receive recommendations.")
        st.write("Select parameters below and view results.")

    elif option == 'Dataset Overview':
        st.header("Dataset Overview")
        df = load_data()
        if st.sidebar.checkbox("Show raw data", key='disp_data'):
            st.write(df)
        st.markdown("### Features Description:")
        st.write("**Course URL:** URL to the course homepage")
        st.write("**Course Name:** Name of the course")
        st.write("**Learning Product Type:** Course, Professional Certificate, or Specialization")
        st.write("**Course Provided By:** Partner providing the course")
        st.write("**Course Rating:** Overall rating of the course")
        st.write("**Course Rated By:** Number of learners who rated the course")
        st.write("**Enrolled Student Count:** Number of learners enrolled")
        st.write("**Course Difficulty:** Difficulty level of the course")
        st.write("**Skills:** Relevant skills covered in the course")
        st.write("**Description:** About the course")
        st.write("**Percentage of New Career Starts:** Learners starting a new career after the course")
        st.write("**Percentage of Pay Increase or Promotion:** Learners receiving pay increase or promotion after the course")
        st.write("**Estimated Time to Complete:** Approximate time to complete")
        st.write("**Instructors:** Instructors of the course")

    elif option == 'Content-based Recommendation':
        df = load_data()
        prep_for_cbr(df)

    elif option == 'About':
        st.header("About")
        st.write("This web app is created by Ayush Katre.")
        st.write("The app allows users to explore courses on Coursera, filter them based on their preferences, and receive recommendations.")
        st.write("It is developed using Streamlit, BeautifulSoup, Pandas, and other Python libraries.")


if __name__ == "__main__":
    main()
