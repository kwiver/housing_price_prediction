# import libraries
import pandas as pd
import streamlit as st
import joblib
import numpy as np

# set page config
st.set_page_config(
    page_title = "Jiji Housing Price Prediction",
    page_icon = "🏫",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

@st.cache_resource
def load_model_artifact():
    try:
        model = joblib.load("rf_model_prediction.pkl")
        print("model", model)
        scaler = joblib.load("scaler_features.pkl")
        print("scaler", scaler)
        feature = joblib.load("feature_columns.pkl")
        print("feature", feature)
        label_encoder = joblib.load("label_encoders.pkl")
        print("label_encoder", label_encoder)
        
        return model, scaler, label_encoder, feature
    except FileNotFoundError as e:
        st.error(f"Model file not found {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occured {e}")
        st.stop()
        
@st.cache_data()
def load_dataset():
    try:
        df = pd.read_csv("jiji_housing_cleaned.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Dataset not found {e}")
        st.stop()


# get filtered options
def get_filtered_options(df, state=None, region_name=None, furnishing=None):
    filtered_df = df.copy()
    
    if state:
        filtered_df = filtered_df[filtered_df["state"] == state]
        
    if region_name:
        filtered_df = filtered_df[filtered_df["region_name"] == region_name]
        
    if furnishing:
        filtered_df = filtered_df[filtered_df["furnishing"] == furnishing]
        
    options ={
        "bathroomss": sorted(filtered_df["bathrooms"].unique().tolist(), reverse=True),
        "price_m2s": sorted(filtered_df["price_m2"].unique().tolist(), reverse=True),
        "states": sorted(filtered_df["state"].unique().tolist()),
        "region_names": sorted(filtered_df["region_name"].unique().tolist()) if state else sorted(df["region_name"].unique().tolist()),
        # "furnishings": sorted(filtered_df["furnishing"].unique().tolist()) if state or region_name else sorted(df["furnishing"].unique().tolist()),
        # "transmissions":  sorted(filtered_df["transmission"].unique().tolist()) if make or model or condition else sorted(df["transmission"].unique().tolist())
        "furnishings": sorted(df["furnishing"].unique().tolist()),
 
    }      
    
    return options


def predict_price(house_data, model, scaler, label_encoder, feature_columns):
    
    try:
        
        input_df = pd.DataFrame([house_data])

    
        columns = ["state", "region_name", "furnishing"]
        for col in columns:
            if col in label_encoder:
                try:
                    input_df[col + "_encoded"] = label_encoder[col].transform(input_df[col])
                except Exception as e:
                    st.warning(f"Unknown {col}: {house_data[col]}. Using default encoding")  
                    input_df[col + "_encoded"] = 0
                
        # feature preparing
        feature_dict = {
            "bathrooms": house_data["bathrooms"],
            "price_m2": house_data["price_m2"],
            "state_encoded": input_df["state_encoded"].values[0],
            "region_name_encoded": input_df["region_name_encoded"].values[0],
            "furnishing_encoded": input_df["furnishing_encoded"].values[0],
            # "transmission_encoded": input_df["transmission_encoded"].values[0],
        }
    
        # create feature array
        features = np.array([[feature_dict[col] for col in feature_columns]])

        # scaler features
        scaler_features  = scaler.transform(features)

        # model prediction
        predicted_price = model.predict(scaler_features)[0]

        margin_percentage = 0.15
        min_predicted_price = predicted_price * (1 - margin_percentage)
        max_predicted_price = predicted_price * (1 + margin_percentage)
        
        return {
        "predicted_price": predicted_price,
        "min_predicted_price": min_predicted_price,
        "max_predicted_price": max_predicted_price  
        }
        
    except Exception as e:
        st.error(f"An error occured: {e}")



def main():
    # header
    st.title("🚘Housing Price Evaluation")
    st.write("Fill all the fields and get immediate result.")
    
    # load model and dataset
    model, scaler, label_encoder, feature_columns = load_model_artifact()
    df = load_dataset()
    
    # initialize session_state
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = None
        
    if "selected_region_name" not in st.session_state:
        st.session_state.selected_region_name = None
    
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
    
    
    # get filter option
    options = get_filtered_options(df)
    
    state = st.selectbox(
        "State*",
        options=[""] + options["states"],
        format_func=lambda  x: "Select State" if x == "" else x
    )
    # update make option
    if state and state != st.session_state.selected_state:
        st.session_state.selected_state = state
        st.session_state.selected_region_name = None
        st.session_state.show_result = False   
    if state:
        options = get_filtered_options(df, state=state)
        
    
    #  select region_name
    region_name = st.selectbox(
         "Region name*",
        options=[""] + options["region_names"],
        format_func=lambda  x: "Select Region Name" if x == "" else x,
        disabled=not state
    )
    # update region option
    if region_name and region_name != st.session_state.selected_region_name:
        st.session_state.selected_region_name = region_name
        st.session_state.show_result = False
    if state and region_name:
        options = get_filtered_options(df, state=state, region_name=region_name)
        
        
     # select price_m2
    price_m2= st.selectbox(
        "Price per square meter*",
        options=[""] + options["price_m2s"],
        format_func=lambda  x: "Select Price Per Square Meter" if x == "" else str(x),
        disabled=not (state and region_name)
    )
    if price_m2:
        st.session_state.show_result = False
    
    
    # select bathrooms
    # bathrooms= st.selectbox(
    #     "Number of bathrooms*",
    #     options=[""] + options["bathroomss"],
    #     format_func=lambda  x: "Select Number of Bathrooms" if x == "" else str(x),
    #     disabled=not (state and region_name and price_m2)
    # )
    bathrooms = st.number_input(
    "Number of bathrooms*",
    min_value=1,
    step=1,
    value=1,
    disabled=not (state and region_name and price_m2)
    )
    if bathrooms:
        st.session_state.show_result = False
        
    
    # select furnishing
    furnishing = st.selectbox(
        "Furnishing*",
        options=[""] + options["furnishings"],
        format_func=lambda  x: "Select Furnishing" if x == "" else x,
        disabled=not (state and region_name and price_m2 and bathrooms)
    )
    if furnishing:
        st.session_state.show_result = False
        
        
    # get result
    if st.button("GET RESULT"):
        if not all([state, region_name, price_m2, bathrooms, furnishing]):
            st.warning("⚠️ Please fill all the firlds")
        else:
            house_data = {
                "state": state,
                "region_name": region_name,
                "price_m2": price_m2,
                "bathrooms": bathrooms,
                "furnishing": furnishing
            }
            with st. spinner("Calculating predicted price..."):
                result = predict_price(house_data, model, scaler, label_encoder, feature_columns)
                
            if result:
                st.session_state.show_result = True
                st.session_state.result = result
                st.session_state.house_data =house_data
    
    # display result
    if st.session_state.show_result and "result" in st.session_state:
        result = st.session_state.result
        house_data = st.session_state.house_data
        
        # display car detail
        st.markdown("---")
        st.subheader(
            f"Estimated house price : ₦{house_data['price_m2']:,.0f} per sq meter, "
            f"{house_data['state']}, {house_data['region_name']}"
            )
        # st.subheader(f"Estimated house price : ₦{house_data["price_m2"]:,.0f} per sq meter, {house_data["state"]}, {house_data["region_name"]}")
        
        # predicted price
        st.success(f"## ₦{result["predicted_price"]:,.0f}")
        
        # price range
        st.write(f"**Price range** ₦{result["min_predicted_price"]:,.0f} - ₦{result["max_predicted_price"]:,.0f}")
        
        # dispaly car detail
        st.write(f"**Furnishing :** {house_data["furnishing"]} | **Bathrooms :** {house_data["bathrooms"]}")
    
    
if __name__ == "__main__":
    main()
    