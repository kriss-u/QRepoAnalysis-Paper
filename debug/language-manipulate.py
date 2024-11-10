import pandas as pd

df = pd.read_csv(
    "data/raw/repo_languages.csv",
    names=["repo", "language", "frameworks", "files", "code", "comments", "blanks"],
)
df_unique = df.loc[
    df.groupby("repo")["frameworks"].apply(lambda x: x.str.count(";").idxmax())
]
df_unique.to_csv("data/raw/repo_languages.csv", index=False)
