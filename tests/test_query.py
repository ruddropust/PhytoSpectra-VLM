from app.query import ConversationContext, plan_query


def test_generalized_single_and_multi_target_queries():
    assert plan_query("Tell me its fertilizer.").targets == ["Fertilizer"]
    assert plan_query("Can you give me the GrainWeight?").intent == "TARGET_VALUE"
    assert plan_query("What is the predicted photosynthetic efficiency?").targets == ["PhiPS2"]
    assert plan_query("Give the two categorical predictions.").targets == ["Fertilizer", "Cultivar"]
    assert plan_query("What are the three continuous measurements?").targets == ["GrainWeight", "Gsw", "PhiPS2"]
    assert plan_query("GRAINWIEGHT value???").targets == ["GrainWeight"]


def test_order_exclusion_and_followup_context():
    assert plan_query("Report Cultivar first, then Gsw, then GrainWeight.").targets == ["Cultivar", "Gsw", "GrainWeight"]
    assert plan_query("Give GrainWeight and Gsw, but do not report Fertilizer.").targets == ["GrainWeight", "Gsw"]
    context = ConversationContext(["Gsw"], "KNOWLEDGE")
    assert plan_query("Show its evidence.", context).targets == ["Gsw"]


def test_natural_paraphrases_and_typographical_errors():
    cases = {
        "Tell me its fertilizer.": ["Fertilizer"],
        "Can you give me the GrainWeight?": ["GrainWeight"],
        "What does the model think the cultivar is?": ["Cultivar"],
        "What are the three continuous measurements?": ["GrainWeight", "Gsw", "PhiPS2"],
        "plz giv fertiliser n cultivar": ["Fertilizer", "Cultivar"],
    }
    for question, expected in cases.items():
        plan = plan_query(question)
        assert plan.include_prediction
        assert plan.targets == expected
