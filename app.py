            ]
        )
        marital = st.selectbox(
            "Marital status",
            ["Unknown", "Single", "Married", "Divorced"]
        )

    with c3:
        income = st.selectbox(
            "Income category",
            [
                "Unknown", "Less than $40K", "$40K - $60K",
                "$60K - $80K", "$80K - $120K", "$120K +"
            ]
        )
        card = st.selectbox(
            "Card category", ["Blue", "Silver", "Gold", "Platinum"]
        )

    st.divider()
    st.markdown("#### 02 — Account Relationship")

    a1, a2, a3 = st.columns(3)

    with a1:
        months_on_book = st.number_input(
            "Months on book", 0, 100, 36
        )
        relationships = st.number_input(
            "Total relationship count", 0, 10, 4
        )

    with a2:
        inactive = st.number_input(
            "Months inactive in last 12 months", 0, 12, 2
        )
        contacts = st.number_input(
            "Contacts in last 12 months", 0, 20, 2
        )

    with a3:
        credit_limit = st.number_input(
            "Credit limit ($)", min_value=0.0,
            value=8000.0, step=500.0
        )
        revolving = st.number_input(
            "Total revolving balance ($)", min_value=0.0,
            value=1200.0, step=100.0
        )
        open_to_buy = st.number_input(
            "Average open to buy ($)", min_value=0.0,
            value=6800.0, step=100.0
        )

    st.divider()
    st.markdown("#### 03 — Transaction Behaviour")

    t1, t2, t3 = st.columns(3)

    with t1:
        amount_change = st.number_input(
            "Transaction amount change (Q4/Q1)",
            min_value=0.0, value=0.7, step=0.1,
            format="%.4f"
        )
        transaction_amount = st.number_input(
            "Total transaction amount ($)",
            min_value=0.0, value=3000.0, step=100.0
        )

    with t2:
        transaction_count = st.number_input(
            "Total transaction count",
            min_value=0, value=45, step=1
        )
        count_change = st.number_input(
            "Transaction count change (Q4/Q1)",
            min_value=0.0, value=0.6, step=0.1,
            format="%.4f"
        )

    with t3:
        utilization = st.number_input(
            "Average utilization ratio",
            min_value=0.0, max_value=1.0,
            value=0.2, step=0.05, format="%.2f"
        )
        st.caption(
            "Utilization ratio ranges from 0 to 1."
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨  Analyze Customer Attrition Risk",
        use_container_width=True
    )


# =========================================================
# PREDICTION RESULTS
# =========================================================
if submitted:

    customer = pd.DataFrame([{
        "Customer_Age": age,
        "Gender": gender,
        "Dependent_count": dependents,
        "Education_Level": education,
        "Marital_Status": marital,
        "Income_Category": income,
        "Card_Category": card,
        "Months_on_book": months_on_book,
        "Total_Relationship_Count": relationships,
        "Months_Inactive_12_mon": inactive,
        "Contacts_Count_12_mon": contacts,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving,
        "Avg_Open_To_Buy": open_to_buy,
        "Total_Amt_Chng_Q4_Q1": amount_change,
        "Total_Trans_Amt": transaction_amount,
        "Total_Trans_Ct": transaction_count,
        "Total_Ct_Chng_Q4_Q1": count_change,
        "Avg_Utilization_Ratio": utilization
    }])

    try:
        prediction = int(model.predict(customer)[0])
        probabilities = model.predict_proba(customer)[0]

        # Assumes class 1 represents Attrited Customer.
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        st.divider()
        st.markdown(
            '<div class="section-title">📈 Prediction Results</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns([1.15, 1])

        with r1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Predicted Customer Status**")

            if prediction == 1:
                st.error("⚠️ Attrited Customer")
                st.markdown(
                    "The model classifies this customer as belonging "
                    "to the attrited class."
                )
            else:
                st.success("✓ Existing Customer")
                st.markdown(
                    "The model classifies this customer as belonging "
                    "to the existing customer class."
                )
