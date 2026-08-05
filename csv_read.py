import pandas as pd

df = pd.read_csv(r'D:\KMAR-50K\KMAR-50K\TrainingCohort.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(5))


df2 = pd.read_csv(r'D:\KMAR-50K\KMAR-50K\TestingCohort.csv')
print(df2.shape)
print(df2.columns.tolist())
print(df2.head(5))


# Check unique values in key columns
for col in ['SeriesDescription', 'Manufacturer', 'ManufacturerModel', 'MagneticFieldStrength']:
    print(f"\n{col}:", df[col].unique())