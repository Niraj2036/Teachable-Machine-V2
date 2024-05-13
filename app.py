import streamlit as st

def main():
    st.title("Teachable Machine V2")
    st.write("Train the model of")

    # Classification and Regression buttons with embedded URLs
    col1, col2 = st.columns(2)
    with col1:
        
        st.write('<a href="https://classification-teachable-machine.streamlit.app" style="text-decoration: none; color: inherit;"><button style="padding: 10px 20px; background-color: #008CBA; color: white; border: none; border-radius: 5px; cursor: pointer;">Classification</button></a>', unsafe_allow_html=True)
    with col2:
        
        st.write('<a href="https://regression-teachable-machine.streamlit.app" style="text-decoration: none; color: inherit;"><button style="padding: 10px 20px; background-color: #008CBA; color: white; border: none; border-radius: 5px; cursor: pointer;">Regression</button></a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
