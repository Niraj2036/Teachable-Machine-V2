import streamlit as st

def main():
    st.title("Teachable Machine V2")
    st.write("Train the model of")

    # Button container
    col1, col2 = st.columns(2)

    # Classification button
    with col1:
        st.markdown("[Classification](https://classification-teachable-machine.streamlit.app)")

    # Regression button
    with col2:
        st.markdown("[Regression](https://regression-teachable-machine.streamlit.app)")

if __name__ == "__main__":
    main()
