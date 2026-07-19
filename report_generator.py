from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_report(prediction, importance):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    #########################################################
    # Title
    #########################################################

    story.append(
        Paragraph(
            "<b>NASA C-MAPSS Predictive Maintenance Report</b>",
            styles["Title"]
        )
    )

    story.append(Spacer(1,20))

    #########################################################
    # Prediction Summary
    #########################################################

    story.append(
        Paragraph(
            "<b>Prediction Summary</b>",
            styles["Heading2"]
        )
    )

    summary = [

        ["Predicted RUL",
         f"{prediction['Predicted_RUL']:.2f} Cycles"],

        ["Estimated Failure",
         str(prediction["Failure_Cycle"])],

        ["Confidence",
         f"{prediction['Confidence']}%"],

        ["Health",
         prediction["Health"]]

    ]

    table = Table(summary)

    table.setStyle(

        TableStyle([

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),

            ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ])

    )

    story.append(table)

    story.append(Spacer(1,20))

    #########################################################
    # SHAP
    #########################################################

    story.append(
        Paragraph(
            "<b>Top Contributing Features</b>",
            styles["Heading2"]
        )
    )

    data = [["Feature","SHAP Value"]]

    for _, row in importance.iterrows():

        data.append(

            [

                row["Feature"],

                round(row["SHAP"],4)

            ]

        )

    shap_table = Table(data)

    shap_table.setStyle(

        TableStyle([

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,0),(-1,0),colors.grey),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ])

    )

    story.append(shap_table)

    story.append(Spacer(1,20))

    #########################################################
    # Footer
    #########################################################

    story.append(

        Paragraph(

            "Generated using XGBoost + SHAP Explainable AI",

            styles["Italic"]

        )

    )

    doc.build(story)

    buffer.seek(0)

    return buffer