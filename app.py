import requests
import streamlit as st

st.set_page_config(page_title="Transcription Audio", layout="centered")
st.title("Transcription Audio vers du texte")

if "image" not in st.session_state:
    st.session_state.image = None

with st.form("audio_form"):
    image_description = st.audio_input(
        "Décrivez l'image que vous souhaitez générer", sample_rate=48000
    )
    submitted = st.form_submit_button("Générer")

    if submitted:
        if image_description is not None:
            try:
                with st.spinner("Génération de l'image en cours..."):
                    response = requests.post(
                        "http://localhost:8000/generate_image",
                        files={
                            "description": (
                                "description.wav",
                                image_description.getvalue(),
                                "audio/wav",
                            )
                        },
                    )
                    if response.status_code == 200:
                        st.session_state.image = response.content
                    else:
                        st.error(
                            f"Erreur lors de la génération de l'image : {response.status_code}"
                        )
            except Exception as e:
                st.error(f"Une erreur s'est produite : {str(e)}")

if st.session_state.image is not None:
    st.image(
        st.session_state.image, caption="Image générée à partir de la description audio"
    )
