import streamlit as st
import pickle

KNN = pickle.load(open('KNN.pkl','rb'))
SVC = pickle.load(open('SVC.pkl','rb'))
rfc = pickle.load(open('rfc.pkl','rb'))

def classify(num):
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'
        
def main():
    st.title("Flower Prediction")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Activity = ['KNeighbors Classifier','Support Vector Classifier','Random Forest Classifier']
    option = st.selectbox("Which model do you want to use?",Activity)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    input = [[sl,sw,pl,pw]]
    if st.button('Predict'):
        if option == 'KNeighbors Classifier':
            st.success(classify(KNN.predict(input)))
        elif option == 'Support Vector Classifier':
            st.success(classify(SVC.predict(input)))
        else:
            st.success(classify(rfc.predict(input)))
            
if __name__=='__main__':
    main()
            