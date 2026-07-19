REQUIRED_COLUMNS = [

    "engine_id",

    "cycle",

    "op_setting_1",

    "op_setting_2",

    "op_setting_3",

]

REQUIRED_COLUMNS.extend(

    [f"sensor_{i}" for i in range(1,22)]

)


def validate(df):

    missing = [

        col

        for col in REQUIRED_COLUMNS

        if col not in df.columns

    ]

    return missing