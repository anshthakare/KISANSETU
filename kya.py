import logging  
import streamlit as st  # Make sure to import Streamlit  

# Set logging level to CRITICAL for the entire app  
logging.basicConfig(level=logging.CRITICAL)  

# Set page configuration at the top  
st.set_page_config(layout="wide")  

# Initialize session state variables  
if "logged_in" not in st.session_state:  
    st.session_state.logged_in = False  
if "username" not in st.session_state:  
    st.session_state.username = ""  
if "prediction" not in st.session_state:  # Initialize prediction  
    st.session_state.prediction = None  
if "prediction_made" not in st.session_state:  # Initialize prediction_made  
    st.session_state.prediction_made = False  

# Your other Streamlit code goes here...
# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Hashing function for security
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Function to add a new user
def add_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()
    conn.close()

# Function to verify user credentials
def verify_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

# Initialize the database
init_db()

# Function to clear session state for logout
def logout():
    st.session_state.logged_in = False  # Set logged_in to False
    st.session_state.username = ""       # Clear the username
    st.session_state.prediction = None    # Clear prediction state
    st.session_state.prediction_made = False  # Clear prediction made state
    st.rerun()  # Rerun the app to show the login page




def display_farming_techniques(predicted_crops):
    techniques_output = {}
    for crop in predicted_crops:
        techniques = farming_techniques.get(crop.lower(), None)
        if techniques:
            techniques_output[crop] = techniques
        else:
            techniques_output[crop] = "Farming techniques not available for this crop."
    return techniques_output

# Sidebar login/signup selection
if not st.session_state.get("logged_in"):
    auth_choice = st.sidebar.selectbox("Choose Authentication", ["Login", "Sign Up"])

    # Form for login or signup
    st.sidebar.write(f"Please {auth_choice} to continue:")
    username_input = st.sidebar.text_input("Username", key="username_input")
    password_input = st.sidebar.text_input("Password", type="password", key="password_input")

    if auth_choice == "Sign Up":
        if st.sidebar.button("Create Account"):
            if username_input and password_input:
                try:
                    add_user(username_input, password_input)
                    st.sidebar.success("Account created successfully! Please log in.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("Username already exists. Please choose a different username.")
            else:
                st.sidebar.warning("Please fill out both fields.")

    elif auth_choice == "Login":
        if st.sidebar.button("Login"):
            user = verify_user(username_input, password_input)
            if user:
                st.sidebar.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.rerun()  # Rerun the app to show the main content
            else:
                st.sidebar.error("Invalid username or password.")

# Only show the main app if the user is logged in
if st.session_state.get("logged_in"):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('file:///{"D:/115455-704757069_small.gif"}');
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write(f"Welcome, {st.session_state['username']}!")

    # Initialize additional session state variables for the main app
    if 'predictions' not in st.session_state:  # Removed extra space in 'predictions '
        st.session_state['predictions'] = None
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False

    # Lottie animation function
    def render_lottie_animation(lottie_url, width="300px", height="300px"):
        lottie_html = f"""
        <div style="display: flex; justify-content: center;">
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player 
                src="{lottie_url}"
                background="transparent" 
                speed="1" 
                style="width: {width}; height: {height}" 
                loop 
                autoplay>
            </lottie-player>
        </div>
        """
        components.html(lottie_html, height=int(height.replace("px", "")))

    # Translation functions
    def get_translator(language_code):
        try:
            return Translator(to_lang=language_code)
        except Exception as e:
            st.error(f"Translation service initialization failed: {str(e)}")
            return None

    def translate(text, translator):
        if translator is None:
            return text
        try:
            return translator.translate(text) or text
        except Exception:
            return text

    # Load dataset with caching
    @st.cache_data
    def load_data():
        dataset_path = r"C:\Users\ANSH\Desktop\Crop_recommendation.csv"  # Adjust this path as needed
        if os.path.exists(dataset_path):
            return pd.read_csv(dataset_path)
        else:
            st.error("Dataset file not found. Please check if 'Crop_recommendation.csv' exists in your project directory.")
            return None

    # Model training with multiple classifiers
    @st.cache_resource
    def train_models_with_augmentation(df):
        if df is None:
            return None, None, {}

        # Preprocessing and augmentation
        X = df.drop('label', axis=1)
        y = df['label']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Define models to train
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'KMeans': KMeans(n_clusters=len(np.unique(y_encoded)), random_state=42)  # KMeans for clustering
        }

        # Dictionary to hold model accuracies and trained models
        accuracies = {}
        trained_models = {}

        # Train each model and calculate accuracy
        for model_name, model in models.items():
            if model_name == 'KMeans':
                model.fit(X_resampled)  # KMeans doesn't use y for fitting in unsupervised learning
                accuracies[model_name] = cross_val_score(RandomForestClassifier(), X_resampled, y_resampled, cv=5).mean()  # Use RandomForest as a baseline score
            else:
                model.fit(X_resampled, y_resampled)
                score = cross_val_score(model, X_resampled, y_resampled, cv=5).mean()
                accuracies[model_name] = score
                trained_models[model_name] = model

        return trained_models, label_encoder, accuracies

    # Language selection
    languages = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
    language_selection = st.sidebar.selectbox("Select Language", list(languages.keys()))
    translator = get_translator(languages[language_selection])

    # Translate static text
    title = translate("KISAN SETU", translator)
    welcome = translate("WELCOME", translator)
    description = translate("Queries are everywhere, solution is here", translator)
    name_input_label = translate("Enter your name:", translator)

    # Layout with two columns
    col1, col2 = st.columns(2)

    # Title and description in the first column
    with col1:
        st.markdown(f"# {title}")
        st.title(welcome)
        st.markdown(f"## {description}")

    # Display logo image in the second column
    with col2:
        image_path = r"C:\Users\ANSH\Desktop\ansh.png.jpg"  # Adjust this path as needed
        if os.path.isfile(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, width=200, use_column_width=False)
            except Exception as e:
                st.warning(f"Error loading image: {str(e)}")
        else:
            st.warning("Logo image not found. Please check if 'ansh.png' exists at the specified path.")

    # Load data and train models
    df = load_data()
    if df is not None:
        models, label_encoder, accuracies = train_models_with_augmentation(df)

        # Display the accuracy of each algorithm
        st.write("### Model Accuracies")
        for model_name, accuracy in accuracies.items():
            st.write(f"{model_name}: {accuracy:.2f}")

        # Dropdown for model selection
        model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))

        # User inputs
        st.text_input(name_input_label)
        st.text("\n")

        # Input fields for parameters with translated labels
        phosphorus = st.number_input(translate("Phosphorus", translator), min_value=0)
        humidity = st.number_input(translate("Humidity", translator), min_value=0.0)
        nitrogen = st.number_input(translate("Nitrogen", translator), min_value=0)
        potassium = st.number_input(translate("Potassium", translator), min_value=0)
        temperature = st.number_input(translate("Temperature (in degree C)", translator), min_value=0.0)
        rainfall = st.number_input(translate("Rainfall (in mm)", translator), min_value=0.0)
        ph_value = st.number_input(translate("pH", translator), min_value=0.0)

        # Farming techniques data for all specified crops
        farming_techniques = {
            "rice": {
                "sowing": "Sow seeds in wet soil or transplant seedlings after 25-30 days.",
                "maintenance": "Maintain flooded fields during most growth stages.",
                "harvesting": "Harvest when grains turn golden and panicles bend.",
                "fertilizers": "Use urea and phosphorus fertilizers at various stages.",
                "max_yield": "Choose high-yield varieties, maintain proper spacing, and control pests.",
                "avoid": "Avoid poor-quality seeds and water shortages."
            },
            "maize": {
                "sowing": "Plant seeds 4-5 cm deep with spacing of 20-25 cm.",
                "maintenance": "Regular watering is essential, especially in dry periods.",
                "harvesting": "Harvest when the husks are dry, and kernels are hard.",
                "fertilizers": "Use a balanced NPK fertilizer mix during growth stages.",
                "max_yield": "Use hybrid seeds, ensure weed control, and optimize irrigation.",
                "avoid": "Avoid delayed planting to prevent pest issues."
            },
            "chickpea": {
                "sowing": "Sow seeds 5-8 cm deep in well-drained soil.",
                "maintenance": "Moderate watering is needed, especially in early growth.",
                "harvesting": "Harvest when pods turn brown and seeds are hard.",
                "fertilizers": "Apply phosphorus fertilizers for better root growth.",
                "max_yield": "Rotate with cereal crops to improve soil health.",
                "avoid": "Avoid waterlogged conditions."
            },
            "kidneybeans": {
                "sowing": "Sow seeds 2-3 cm deep in well-drained soil.",
                "maintenance": "Water consistently, especially during flowering.",
                "harvesting": "Harvest when pods are firm and dry.",
                "fertilizers": "Use nitrogen-fixing fertilizers sparingly.",
                "max_yield": "Ensure good weed control and use quality seeds.",
                "avoid": "Avoid overwatering and poor drainage."
            },
            "pigeonpeas": {
                "sowing": "Sow seeds 5-7 cm deep with spacing of 20-25 cm.",
                "maintenance": "Minimal watering required after establishment.",
                "harvesting": "Harvest when pods dry and turn brown.",
                "fertilizers": "Apply phosphorus for better yield.",
                "max_yield": "Intercrop with cereals to improve soil structure.",
                "avoid": "Avoid planting in waterlogged soils."
            },
            "mothbeans": {
                "sowing": "Sow seeds 2-3 cm deep in sandy soil.",
                "maintenance": "Requires minimal watering after germination.",
                "harvesting": "Harvest when pods turn yellow and dry.",
                "fertilizers": "Minimal fertilizer needed; use organic manure.",
                "max_yield": "Ensure proper weed control.",
                "avoid": "Avoid excessive irrigation."
            },
            "mungbean": {
                "sowing": "Sow seeds 3-4 cm deep in well-prepared soil.",
                "maintenance": "Water during flowering and pod development.",
                "harvesting": "Harvest when pods turn yellow.",
                "fertilizers": "Use phosphorus fertilizers for better yield.",
                "max_yield": "Practice crop rotation to improve soil health.",
                "avoid": "Avoid overwatering."
            },
            "blackgram": {
                "sowing": "Sow seeds 3-5 cm deep in light soil.",
                "maintenance": "Requires moderate irrigation, especially at flowering.",
                "harvesting": "Harvest when pods mature and turn brown.",
                "fertilizers": "Use phosphorus fertilizers for root growth.",
                "max_yield": "Ensure weed control and avoid water stress.",
                "avoid": "Avoid poor drainage and waterlogging."
            },
            "lentil": {
                "sowing": "Sow seeds 3-4 cm deep with adequate spacing.",
                "maintenance": "Water during flowering and pod formation stages.",
                "harvesting": "Harvest when pods turn yellow and dry.",
                "fertilizers": "Apply phosphorus fertilizers before sowing.",
                "max_yield": "Rotate with cereals for soil improvement.",
                "avoid": "Avoid excessive irrigation."
            },
            "pomegranate": {
                "sowing": "Plant cuttings or grafted plants in well-drained soil.",
                "maintenance": "Water moderately and prune regularly.",
                "harvesting": "Harvest when fruits are mature and color deepens.",
                "fertilizers": "Use potassium-rich fertilizers during fruiting.",
                "max_yield": "Prune to allow sunlight penetration.",
                "avoid": "Avoid waterlogging and high salinity."
            },
            "banana": {
                "sowing": "Plant suckers or tissue-cultured plants in rich, well-drained soil.",
                "maintenance": "Keep soil moist and mulch regularly.",
                "harvesting": "Harvest when bananas reach size but are still green.",
                "fertilizers": "Apply potassium-rich fertilizer during growing season.",
                "max_yield": "Remove dead leaves and sucker control.",
                "avoid": "Avoid waterlogging and excessive pruning."
            },
            "mango": {
                "sowing": "Plant grafted seedlings in well-drained soil.",
                "maintenance": "Water regularly, especially during flowering.",
                "harvesting": "Harvest when fruits are fully grown and change color.",
                "fertilizers": "Use nitrogen and potassium fertilizers.",
                "max_yield": "Prune to improve airflow and sunlight.",
                "avoid": "Avoid water stress and frost exposure."
            },
            "grapes": {
                "sowing": "Plant cuttings in well-drained, sandy loam soil.",
                "maintenance": "Water consistently, especially during dry periods.",
                "harvesting": "Harvest when berries reach desired sweetness.",
                "fertilizers": "Apply nitrogen and potassium.",
                "max_yield": "Prune regularly for good air circulation.",
                "avoid": "Avoid waterlogged soils."
            },
            "watermelon": {
                "sowing": "Sow seeds 2-3 cm deep in well-drained soil.",
                "maintenance": "Water moderately, especially during flowering.",
                "harvesting": "Harvest when fruit sounds hollow when tapped.",
                "fertilizers": "Use nitrogen and potassium-rich fertilizers.",
                "max_yield": "Ensure enough sunlight and spacing.",
                "avoid": "Avoid overwatering."
            },
            "muskmelon": {
                "sowing": "Sow seeds 2-3 cm deep in warm soil.",
                "maintenance": "Water consistently, especially during fruit setting.",
                "harvesting": "Harvest when fruits emit a sweet aroma.",
                "fertilizers": "Use balanced NPK fertilizers.",
                "max_yield": "Ensure good sunlight and weed control.",
                "avoid": "Avoid waterlogging."
            },
            "apple": {
                "sowing": "Plant grafted seedlings in cool, well-drained soil.",
                "maintenance": "Water regularly and prune annually.",
                "harvesting": "Harvest when fruits are firm and fully colored.",
                "fertilizers": "Use nitrogen and potassium.",
                "max_yield": "Prune to allow sunlight.",
                "avoid": "Avoid poor drainage."     
            },
            "orange": {
                "sowing": "Plant grafted seedlings in well-drained soil.",
                "maintenance": "Water regularly, especially in dry periods.",
                "harvesting": "Harvest when fruits are fully colored.",
                "fertilizers": "Use nitrogen-rich fertilizers.",
                "max_yield": "Prune to improve airflow.",
                "avoid": "Avoid waterlogging."
            },
            "papaya": {
                "sowing": "Plant seeds 1-2 cm deep in warm soil.",
                "maintenance": "Water consistently, avoid standing water.",
                "harvesting": "Harvest when fruits change color.",
                "fertil izers": "Use potassium-rich fertilizers.",
                "max_yield": "Prune old leaves.",
                "avoid": "Avoid cold and waterlogged conditions."
            },
            "coconut": {
                "sowing": "Plant nuts in deep, well-drained sandy soil.",
                "maintenance": "Water regularly, especially young plants.",
                "harvesting": "Harvest mature nuts after 12-14 months.",
                "fertilizers": "Use potassium and nitrogen fertilizers.",
                "max_yield": "Ensure spacing for sunlight.",
                "avoid": "Avoid high salt soil."
            },
            "coffee": {
                "sowing": "Plant seeds in shaded, well-drained soil.",
                "maintenance": "Keep soil moist but not waterlogged.",
                "harvesting": "Harvest when berries turn red.",
                "fertilizers": "Use nitrogen fertilizers.",
                "max_yield": "Shade and prune regularly.",
                "avoid": "Avoid extreme temperatures."
            },
            "cotton": {
                "sowing": "Sow seeds 2-3 cm deep in well-drained soil.",
                "maintenance": "Irrigate regularly during flowering.",
                "harvesting": "Harvest when bolls open.",
                "fertilizers": "Use NPK fertilizers.",
                "max_yield": "Control pests and ensure good drainage.",
                "avoid": "Avoid over-fertilization."
            },
            "jute": {
                "sowing": "Sow seeds 2-3 cm deep in clay soil.",
                "maintenance": "Water as required, especially in dry periods.",
                "harvesting": "Harvest when flowers start blooming.",
                "fertilizers": "Use nitrogen and phosphorus.",
                "max_yield": "Ensure proper spacing.",
                "avoid": "Avoid poor drainage."
            }
        }
        techniques_output = {}

        # Prediction button
        if st.button(translate("Predict Crop", translator), key="predict_button"):
            render_lottie_animation("https://lottie.host/e0a7e124-6e66-48e4-a2e8-b041e7b0064f/t4ZGHyHqex.json")
            
            try:
                # Use the selected model for prediction
                model = models[model_choice]
                input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
                probabilities = model.predict_proba(input_data)

                # Get the top 3 predictions
                top_n_indices = np.argsort(probabilities[0])[-3:][::-1]  # Get indices of top 3 probabilities
                top_n_probs = probabilities[0][top_n_indices]  # Get top 3 probabilities
                top_n_labels = label_encoder.inverse_transform(top_n_indices)  # Get corresponding labels

                # Store top predictions in session state
                st.session_state.prediction = [(label, prob) for label, prob in zip(top_n_labels, top_n_probs)]
                st.session_state.prediction_made = True
                logging.info(translate(f"Predictions made: {st.session_state.prediction}",translator))

                # Display predictions
                st.write(translate("### Predicted Crop Recommendations",translator))
                for i, (label, prob) in enumerate(st.session_state.prediction, start=1):
                    st.write(f"{i}. {label} - Probability: {prob:.2%}")

                    with st.expander(translate(f"Farming Techniques for {label.capitalize()}",translator)):
                        # Correctly access the techniques for the crop
                        techniques = farming_techniques.get(label.lower(), "Farming techniques not available for this crop.")
        
                        if isinstance(techniques, dict):
                            for technique, detail in techniques.items():
                                st.write(translate(f"{technique.capitalize()}: {detail}",translator))
                        else:
                            st.write(translate(techniques,translator))

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                logging.error(f"Prediction error: {e}")
                st.session_state.prediction = None
                st.session_state.prediction_made = False

        # Feedback section
        st.markdown("---")
        st.subheader(translate("Feedback", translator))
        st.selectbox(
            translate("Did you like our services?", translator),
            [translate(x, translator) for x in ["Yes", "No", "Maybe"]]
        )
        st.radio(translate("Rating", translator), range(6))

        # Description section
        st.markdown("---")
        description_text = translate("""### Our Vision: Empowering Farmers through Technology
        The KISAN SETU platform is designed to assist farmers in making informed decisions about crop production. 
        By leveraging advanced machine learning techniques, the platform analyzes key environmental factors such as soil nutrients, 
        temperature, humidity, and rainfall, to recommend the most suitable crop for cultivation.
        """, translator)
        st.markdown(description_text)

        # Final progress bar
        st.progress(100)

        # Final Lottie animation
        render_lottie_animation("https://lottie.host/1ca725b0-9a6a-47a2-b079-abb12cbefc42/Qg90bOMKAX.json", width="100%", height="300px")

        # Logout button
        if st.button(translate("Logout",translator)):
            logout()

else:
    st.info("Please log in to access the application.")

    
