import streamlit as st

def main():
    st.title("Teachable Machine V2")
    st.write("Train the model of")

    # Classification and Regression buttons with URLs
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Classification"):
            st.write("Redirecting to Classification App...")
            # Redirect to the classification app URL
            st.experimental_rerun()
        st.markdown("[Go to Classification App](https://classification-teachable-machine.streamlit.app)")
    with col2:
        if st.button("Regression"):
            st.write("Redirecting to Regression App...")
            # Redirect to the regression app URL
            st.experimental_rerun()
        st.markdown("[Go to Regression App](https://regression-teachable-machine.streamlit.app)")

if __name__ == "__main__":
    main()
