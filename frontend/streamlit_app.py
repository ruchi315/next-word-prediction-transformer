import streamlit as st
import requests


st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠"
)


st.title("🧠 Next Word Prediction System")


st.write(
    "Enter a sentence and the model will predict the most likely next words."
)


text = st.text_input(
    "Enter text"
)



if st.button("Predict"):

    if text.strip()=="":

        st.warning(
            "Please enter some text"
        )

    else:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "text":text
            }
        )


        if response.status_code == 200:


            result = response.json()


            st.success(
                "Prediction completed"
            )


            st.subheader(
                "Input Text"
            )

            st.write(
                result["input"]
            )


            st.subheader(
                "Generated Next Words"
            )


            prediction = result["prediction"]


            if len(prediction)>0:

                st.write(
                    " ".join(prediction)
                )

            else:

                st.write(
                    "No prediction generated"
                )


        else:

            st.error(
                response.text
            )


st.divider()

st.caption(
    "Built using Deep Learning + FastAPI + Streamlit"
)