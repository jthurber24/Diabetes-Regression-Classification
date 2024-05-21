#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import re
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 50
pd.options.mode.chained_assignment = None


# In[2]:


df = pd.read_csv("C:\pt_intake.csv")
print(df.shape)


# In[3]:


# View column headers
columns = list(df.columns)
print(columns)


# In[4]:


# Remove columns with: 1) Comment boxes, 2) Time/Minutes, 3) Large % of missing data.
to_drop = ['EncounterDate',
           'MOS',
           'LearningDisabilities',
            'LearningDisabilitiesType',
            'YearOfCurrentInjury',
             'MonthsCurrentInjury',
             'GWOT',
           'MechanismInjury',
           'TimeOfArrival',
           'ReferralReason', 
           'MOS_FunctDescDuties', 
           'NumSchoolYrs',
           'PastHistory', 
           'AcquiredBrainInjury',
         'NeurologicalDisease',
         'OtherDiagnosis',
           'RecentHistory', 
           'FacialSkullFract', 
           'Amputation', 
           'AmpRArmBelowElbow', 
           'AmpRArmAboveElbow', 
           'AmpLeftArmBelowElbow', 
           'AmpLeftArmAboveElbow', 
           'AmpRightLegBelowKnee', 
           'AmpRightLegAboveKnee', 
           'AmpLeftLegBelowKnee', 
           'OtherDiagnosis',
             'PTSD_DX',
             'CT_SLP',
             'CT_OT',
         'CT_PT',
         'CT_BehavHealth',
         'PrevNPTesting',
           'VisitNumber',
           'AmpLeftLegAboveKnee', 
           'NumAmputations',  
           'VisualDeficits', 
           'AuditoryDeficitsType', 
           'CT_SLP_Current', 
           'CT_SLP_Past',
           'CT_OT_Current', 
           'CT_OT_Past', 
           'CT_PT_Current', 
           'CT_PT_Past',
           'CT_BehavHealthCurrent', 
           'CT_BehavHealthPast', 
           'ChiefCogComplaints', 
           'PtsBfcGoals', 
           'PtsExpWithBrainTrain', 
           'PainLocation', 
           'PainRecTreatment', 
           'SubjObservations', 
           'PlanOfCareDiscussed', 
           'PlanOfCareBarriers', 
           'PlanOfCareComments', 
           'ReasonForTerm', 
           'CompletedDuringIntake', 
           'BfcBatteryDates_1', 
           'BfcBatteryDates_2', 
           'BfcBatteryDates_3', 
           'HelpAreaComments', 
           'VisualDeficitsType', 
           'VisualDeficitsLR', 
           'VisualDeficitsEvaluated', 
           'AuditoryDeficitsEvaluated', 
           'AuditoryDeficits', 
           'AuditoryDeficitsLR', 
           'ReferralOther', 
           'PrimaryDxOther', 
           'PsychiatricDxOther', 
           'AcquiredBrainInjuryOther', 
           'NeuroDiseaseDisorderOther', 
           'TimeOfDeparture', 
           'Tinnitus', 
           'PlanOfCareLocation', 
           'ProgramUsedComments', 
           'TBIDXComments', 
           'PTSDDxComments', 
           'MechanismInjuryComments', 
           'FacialSkullFractComments', 
           'ResearchAssistant', 
           'EducationComments',
          'PlanOfCare',
         'HelpedAreas'     
          ]

df.drop(to_drop, inplace=True, axis=1)


# In[5]:


# Create column for number of evals
df['NumberOfEvals'] = df.groupby('pseudo_id')['pseudo_id'].transform('count')

# Rearrange df
column_to_move = df.pop("NumberOfEvals")
df.insert(1, "NumberOfEvals", column_to_move)


# In[6]:


# Print current columns
print(df.columns)


# <font color=red>**CLEAN AND CHECK VARIABLES OF INTEREST, THEN DROP ROWS OF ZERO INTEREST** <font>
# 1. Independent Variables: 
#        'pseudo_id', 'Location', 'Gender', 'Service', 'Rank',
#        'Status', 'MaritalStatus', 'Education', 'TBI_DX', 'Severity',
#        'PrimaryDX', 'PsychiatricDX', 'Comorbid_PSYC_DX_Other_Psych_dx',
#        'ConfidenceRating', 'PainScale', 'Evaluation', 'ProgramUsed',
#        'PatientAge'
#     
# 2. Dependent Variable: 'Number of Evals' -> 2 groups: 
#     - 0: Only 1 evaluation
#     - 1: >= 2 evaluations
#     
#     
# 3. Check: 
#     - ReferralSource: to remove populations that do not return for F/U due to inpatient: 7E, IOP, 4 WEEK
#     
#     
# 4. Population: 
#     - AD with TBI

# In[7]:


# Check and remove IOP rows
print(df['ReferralSource'].unique())

# Drop rows that are: IOP, NIcoE 4 week, 7 east
df = df.drop(df.index[df['ReferralSource'].isin(['NICoE 4week', '7-East', 'IOP'])])


# In[8]:


# Clean status column and keep patients that are AD
# 1. Check unique values 
print(df['Status'].unique()) 
# 2. Standardize ['AD' 'RET' nan 'AD Res' 'Dependent' 'NG' 'Unknown' '(none)' 'ad' 'ADRes']
df['Status'] = df['Status'].str.replace(r'\bad\b', 'AD', regex = True)
df['Status'] = df['Status'].str.replace(r'\bADRes\b', 'AD Res', regex = True)
df['Status'] = df['Status'].str.replace(r'\(none\)', 'Other', regex = True)
df['Status'] = df['Status'].str.replace(r'\bUnknown\b', 'Other', regex = True)
print(df['Status'].unique()) 


# In[9]:


# Check numbers for status 
test1 = df.groupby(['Status']).size().to_frame(name = 'count').reset_index()
display(test1)


# In[10]:


# Drop all rows and keep only patients that are AD since good sample size
df = df.query("Status == 'AD'")


# In[11]:


# Clean Primary DX column (alternative use mapping)
# 1. Check unique values 
print(df['PrimaryDX'].unique()) 
# 2. Standardize ['TBI' 'ABI' 'PSYC' nan 'Neurological Disease/Disorder' 'Other' 'Unknown' '(none)' 'Neurological Disease' 'Other (Cognitive complaints)' 'Other (Auditory Processing)' 'Other (Multiple Sclerosis)' 'Other '  'Other (Cervical Spine Surgery)']
# Remove backslash and parenthesis
df.replace('/','', regex=True, inplace=True)
df['PrimaryDX'].str.replace(r"\(.*\)","", regex=True)

df['PrimaryDX'] = df['PrimaryDX'].str.replace(r'\bNeurological DiseaseDisorder\b', 'Neurological Disorder', regex = True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace(r'\bNeurological Disorder\b', 'Neurological Disease', regex = True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace('Other \(Multiple Sclerosis\)', 'Other', regex=True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace('Other \(Cognitive complaints\)', 'Other', regex=True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace('Other \(Cervical Spine Surgery\)', 'Other', regex=True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace('Other \(Auditory Processing\)', 'Other', regex=True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace('Other\s+', 'Other', regex = True)

df['PrimaryDX'] = df['PrimaryDX'].str.replace(r'\(none\)', 'Other', regex = True)
df['PrimaryDX'] = df['PrimaryDX'].str.replace(r'\bUnknown\b', 'Other', regex = True)

# df['PrimaryDX'] = df['PrimaryDX'].replace('(none)', np.nan)
print(df['PrimaryDX'].unique()) 


# In[12]:


# Check numbers for primary dx
test2 = df.groupby(['PrimaryDX']).size().to_frame(name = 'count').reset_index()
display(test2)


# In[13]:


# Drop all rows and keep only patients that are TBI since good n.
df = df.query("PrimaryDX == 'TBI'")


# In[14]:


# Clean service column 
print(df['Service'].unique())
# ['USA' 'USAF' 'USPHS' nan 'USMC' 'USN' 'USCG' 'Unknown' '(none)' 'usn' 'USN ']
df['Service'] = df['Service'].str.replace('usn', 'USN')
df['Service'] = df['Service'].str.replace('USN ', 'USN')
# df['Service'] = df['Service'].replace('(none)', np.nan)

df['Service'] = df['Service'].str.replace(r'\(none\)', 'Other', regex = True)
df['Service'] = df['Service'].str.replace(r'\bUnknown\b', 'Other', regex = True)
df['Service'] = df['Service'].str.replace(r'\bUSCG\b', 'Other', regex = True)
df['Service'] = df['Service'].str.replace(r'\bUSPHS\b', 'Other', regex = True)
# # Display count for each service
print(df.Service.value_counts())
print("Nan Count: " + str(df['Service'].isna().sum()))


# In[15]:


# Remove rows with unknowns (UCSG, USPHS, none)
df = df[df['Service'].str.contains('Other')==False]


# In[16]:


# Check count for each service
print(df.Service.value_counts())
print("Nan Count: " + str(df['Service'].isna().sum()))


# In[17]:


# Clean TBI_DX 
# 1. Check unique values
print(df['TBI_DX'].unique())
# 2. Standardize ['Yes' 'No' nan 'Unknown' '(none)' 'no']
df['TBI_DX'] = df['TBI_DX'].str.capitalize() # I did capitalize first because if you try to replace 'no' with 'No'...this affects the 'no' in (none) and 'Unknown'
# df['TBI_DX'] = df['TBI_DX'].replace('(none)', np.nan)

# Display count for DX
print(df.TBI_DX.value_counts())
print("Nan Count: " + str(df['TBI_DX'].isna().sum()))


# In[18]:


# Remove rows with unknown or no or nan. 
df = df[df['TBI_DX'].str.contains('No|Unknown')==False]


# In[19]:


# Check if rows for TBI DX with no/unknown have been removed
print(df.TBI_DX.value_counts())
print("Nan Count: " + str(df['TBI_DX'].isna().sum()))


# In[20]:


# Clean severity column
# 1. Check unique values 
print(df['Severity'].unique()) 
# 2. Standardize ['Severe' 'Mild' nan 'Unknown' 'Mod' 'Penetrating' 'Severe, Penetrating' '(none)' 'mild' 'Moderate']. Used TBI definition handout to classify to: mild, moderate, severe, penetrating. 
df['Severity'] = df['Severity'].str.replace(r'\bMod\b', 'Moderate', regex = True)
df['Severity'] = df['Severity'].str.replace('mild', 'Mild')
df['Severity'] = df['Severity'].str.replace('Severe, Penetrating', 'Penetrating')
df['Severity'] = df['Severity'].str.replace(r'\(none\)', 'Other', regex = True)
df['Severity'] = df['Severity'].str.replace(r'\bUnknown\b', 'Other', regex = True)
# df_sub[‘Severity’] = df_sub[‘Severity’].replace({‘mild’: ‘Mild’,‘Mod’:‘Moderate’,np.nan:‘Other/Unknown’})

# # Display count 
print(df.Severity.value_counts())
print("Nan Count: " + str(df['Severity'].isna().sum()))


# In[21]:


# Remove rows with unknown or no or nan. 
df = df[df['Severity'].str.contains('Other')==False]

# Check if rows with no/unknown have been removed
print(df.Severity.value_counts())
print("Nan Count: " + str(df['Severity'].isna().sum()))


# In[22]:


# Clean gender column
# 1. Check unique values 
print(df['Gender'].unique()) 
# 2. Standardize ['Male' 'Female' nan '(none)' 'female' 'male']
df['Gender'] = df['Gender'].str.replace(r'\bfemale\b', 'Female', regex = True)
df['Gender'] = df['Gender'].str.replace(r'\bmale\b', 'Male', regex = True)


# In[23]:


# Clean marital status column. Other includes: divorced, separated, widowed
# 1. Check unique values 
# print(df['MaritalStatus'].unique()) 
# 2. Standardize ['Married' 'Single' nan 'Unknown' 'Divorced' 'Widowed' 'Separated''Engaged' '(none)' 'married']
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bmarried\b', 'Married', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bSingle\b', 'Never Married', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bEngaged\b', 'Never Married', regex = True)
# df['MaritalStatus'] = df['MaritalStatus'].replace('(none)', np.nan)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\(none\)', 'Other', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bUnknown\b', 'Other', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bDivorced\b', 'Other', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bWidowed\b', 'Other', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].str.replace(r'\bSeparated\b', 'Other', regex = True)
df['MaritalStatus'] = df['MaritalStatus'].fillna('Other')

# print(df['MaritalStatus'].unique()) 

# Display count 
print(df.MaritalStatus.value_counts())
print("Nan Count: " + str(df['MaritalStatus'].isna().sum()))


# In[24]:


# Remove rows with unknown or no or nan. 
# df = df[df['MaritalStatus'].str.contains('No|Unknown')==False
df = df[df['MaritalStatus'].notna()]

# Check count 
print(df.MaritalStatus.value_counts())
print("Nan Count: " + str(df['MaritalStatus'].isna().sum()))


# In[25]:


# Clean education column. HIGHEST LEVEL OF EDUCATION ONLY. 
# 1. Check unique values 
# print(df['Education'].unique()) 
# 2. Standardize ['Bachelors' 'Some Bachelors' 'Advanced' 'Masters' nan 'High School' 'Associates' 'Unknown' 'Some High School' 'Technical' 'Other' 'High school']
# Replace apostrophes in df ** IMPORTANT FOR REGEX AFTER SEEING THAT THERE ARE APOSTROPHES
df.replace('\'','', regex=True, inplace=True) 
df['Education'] = df['Education'].str.title()
df['Education'] = df['Education'].str.replace(r'Degree', '', regex = True)
df['Education'] = df['Education'].str.replace(r'\bMasters & Advanced\b|Masters|Advanced', 'Masters & Advanced', regex = True)
df['Education'] = df['Education'].str.replace(r'College', 'Bachelors', regex = True)
df['Education'] = df['Education'].str.replace(r'Some Bachelors', 'Some College', regex = True)
df['Education'] = df['Education'].str.replace(r'Ged', 'High School', regex = True)
df['Education'] = df['Education'].str.replace(r'\(None\)', 'Other', regex = True)
df['Education'] = df['Education'].str.replace(r'\bUnknown\b', 'Other', regex = True)
df['Education'] = df['Education'].str.replace(r'\bSome High School\b', 'High School', regex = True)
df['Education'] = df['Education'].str.strip()

# Display count 
print(df.Education.value_counts())
# print("Nan Count: " + str(df['Education'].isna().sum()))


# In[26]:


# Remove rows with unknown or no or nan or other.
df = df[df['Education'].str.contains('Other')==False] # Dropped others since AD SMs require minimum HS education. 
df = df[df['Education'].notna()]

# Check count 
print(df.Education.value_counts())
print("Nan Count: " + str(df['Education'].isna().sum()))


# In[27]:


# Clean rank column
# 1. Check unique values 
print(df['Rank'].unique()) 
# 2. Standardize ['O3' 'E7' 'E5' 'E4' 'E1' 'E3' 'E9' 'O5' 'Unknown' 'O2' 'E6' 'W2' 'O4' nan 'W5' 'E8' 'O6' 'E2' 'W3' 'O1' 'O7' 'W4' 'e6' 'O8' 'Cadet' '(none)']
# Junior enlisted (E1 - E5), senior enlisted (E6-E9), all W/O officers.
df['Rank'] = df['Rank'].replace(dict.fromkeys(['E1','E2','E3','E4', 'E5'], 'Junior Enlisted'))
df['Rank'] = df['Rank'].replace(dict.fromkeys(['E6','e6', 'E7','E8','E9'], 'Senior Enlisted'))
df['Rank'] = df['Rank'].replace(dict.fromkeys(['O3','O4', 'O5','O1','O2', 'O6', 'O7', 'O8', 'W1', 'W2', 'W3', 'W4', 'W5'], 'Officer'))
df['Rank'] = df['Rank'].str.replace(r'\bCadet\b', 'Other', regex = True)

df['Rank'] = df['Rank'].str.replace(r'\bUnknown\b', 'Other', regex = True)
df['Rank'] = df['Rank'].str.replace(r'\(none\)', 'Other', regex = True)
df['Rank']=df['Rank'].fillna(value='Other')

# Display count 
print(df.Rank.value_counts())
print("Nan Count: " + str(df['Rank'].isna().sum()))


# In[28]:


# Remove other
# df = df[df['Rank'].str.contains('Other')==False]
# print(df.Rank.value_counts())


# In[29]:


# Clean confidence column
# 1. Check unique values 
# print(df['ConfidenceRating'].unique()) 
# 2. Standardize ['Unknown' nan '70%' '50%' '75%' '90%' '100%' '40%' '60%' '30%' '80%' '20%' '10%' '85%' '(none)' ' Unknown' '0']
df['ConfidenceRating'] = df['ConfidenceRating'].str.replace(r'\b85\b', '80', regex = True)
df['ConfidenceRating'] = df['ConfidenceRating'].str.replace(r'\b75\b', '70', regex = True)
df['ConfidenceRating'] = df['ConfidenceRating'].replace(' ', '_', regex=True)
df['ConfidenceRating'] = df['ConfidenceRating'].str.replace(r'\b_Unknown\b', 'Unknown', regex = True)
df['ConfidenceRating'] = df['ConfidenceRating'].replace('(none)', np.nan)
print(df['ConfidenceRating'].unique()) 

# Display count 
print(df.ConfidenceRating.value_counts())
print("Nan Count: " + str(df['ConfidenceRating'].isna().sum()))


# In[30]:


# Clean pain column
# 1. Check unique values 
print(df['PainScale'].unique()) 
# 2. Standardize none
df['PainScale'] = df['PainScale'].replace('(none)', np.nan)
# print(df['ConfidenceRating'].unique()) 

# Display count 
print(df.PainScale.value_counts())
print("Nan Count: " + str(df['PainScale'].isna().sum()))


# In[31]:


# Clean age column
# 1. Check unique values 
print(df['PatientAge'].unique()) 

# Display count 
print("Nan Count: " + str(df['PatientAge'].isna().sum()))


# In[32]:


# Remove nan from age 
df = df[df['PatientAge'].notna()]

# Check 
print("Nan Count: " + str(df['PatientAge'].isna().sum()))


# In[33]:


# Clean psychiatric column and code
# Clean rank column
# 1. Check unique values 
print(df['PsychiatricDX'].unique()) 
# DataFrame.replace({'column_name_1' : { old_value_1 : new_value_1, old_value_2 : new_value_2},
# df = df.replace({'a':{1:11, 2:22}})


# 2. Standardize ['Unknown' 'Comorbid' 'PTSD' 'Anxiety' nan 'Depression' 'Bi Polar' 'Adjustment Disorder' 'Comorbid(if comorbid add comment)' 'depression' 'Other' '(none)']
# df=df.replace({'PsychiatricDX': {'Unknown': 0,'Comorbid': 1,'PTSD': 1, 'Anxiety': 1, 'Depression':1,  'Bi Polar': 1,
#                                  'Adjustment Disorder': 1, 'Comorbid(if comorbid add comment)': 1,  'depression': 1, 
#                                  'Other': 1, '(none)': 0}})
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bComorbid\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bPTSD\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bDepression\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bBi Polar\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bAdjustment Disorder\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bdepression\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bOther\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bAnxiety\b', 1, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace(r'\bUnknown\b', 0, regex = True)
df['PsychiatricDX'] = df['PsychiatricDX'].replace('(none)', np.nan)
df['PsychiatricDX'] = df['PsychiatricDX'].fillna(0)

# # # # Display count 
print(df.PsychiatricDX.value_counts())
print("Nan Count: " + str(df['PsychiatricDX'].isna().sum()))


# In[ ]:





# In[34]:


# See spread for age and code age
print(df.PatientAge.value_counts())
print(sorted(df["PatientAge"].unique()))

# df.loc[(df['PatientAge'] >= 20) & (df['PatientAge'] < 30), 'PatientAge'] = 0
# df.loc[(df['PatientAge'] >= 30) & (df['PatientAge'] < 40), 'PatientAge'] = 1
# df.loc[(df['PatientAge'] >= 40) & (df['PatientAge'] < 50), 'PatientAge'] = 2
# df.loc[df['PatientAge'] >= 50, 'PatientAge'] = 3


# In[35]:


print(df.shape) 


# ## Create dfs ##
#  - one with both groups 0 and 1 with groups 1 that consists of only initial evals. 

# In[36]:


# Create column for number of evals
df['Group'] = np.where(df['NumberOfEvals'] >= 2, 1, 0)

# # Rearrange df
column_to_move1 = df.pop("Group")
df.insert(1, "Group", column_to_move1)


# In[37]:


# Patient 1298 has 2 duplicate entries
display((df).loc[df['pseudo_id'] == 1298])


# In[38]:


# Drop duplicate pseudo ID for this subject 1298
# df = df.drop([1077, 1543])
df = df.sort_values('pseudo_id').drop_duplicates(subset=['pseudo_id','Evaluation'], keep='last')


# In[39]:


# Check is 1298 has been dropped
display((df).loc[df['pseudo_id'] == 1298])
print(df.shape) 


# In[40]:


# Patient 499 has 2 duplicate entries
display((df).loc[df['pseudo_id'] == 499])


# In[41]:


# Drop duplicate pseudo ID for this subject 499. Index 1341 should be a RE-EVALUATION but entered incorrectly. 
df = df.drop([1341])


# In[42]:


# Create new df for those with 1+ and 1 eval in case I wanted to examine closer
group1 = df.loc[(df['NumberOfEvals'] > 1) & (df['Evaluation']=='Initial Evaluation')]
# group1.drop_duplicates(subset=['pseudo_id'], keep="first", inplace=True) 
group0 = df.loc[(df['NumberOfEvals'] == 1)  & (df['Evaluation']=='Initial Evaluation')]
print("Group 1: ", group1.shape)  
print("Group 0: ", group0.shape) 


# In[43]:


# Check for dupl
print(group1.pseudo_id.value_counts())


# In[44]:


print(group0.pseudo_id.value_counts())


# In[45]:


# Count unique patients in group 1 and 0
print("Number of patients in group 1: ", len(pd.unique(group1['pseudo_id'])))
print("Number of patients in group 0: ", len(pd.unique(group0['pseudo_id'])))


# In[46]:


print(df['Evaluation'].unique()) 
print(df.Evaluation.value_counts())


# In[47]:


# Create df with only initial evals
descripDf = pd.concat([group1, group0])

print("Total unique patients: ", descripDf.shape) 
print(descripDf['Evaluation'].unique()) 
print(descripDf.Evaluation.value_counts())


# In[48]:


# Recode IVs to be numeric. Use descripDf. 
# print(descripDf['MaritalStatus'].unique())
# print(sorted(descripDf['Education'].unique()))
# print(descripDf.MaritalStatus.value_counts())
# descripDf['Severity'] = descripDf['Severity'].replace({'Mild':0, 'Moderate':1, 'Severe':2, 'Penetrating':3})
# descripDf['Gender'] = descripDf['Gender'].replace({'Male':0, 'Female':1})
# descripDf['Service'] = descripDf['Service'].replace({'USA':0, 'USAF': 1, 'USCG':2, 'USMC':3, 'USN':4, 'USPHS':5})
# descripDf['Education'] = descripDf['Education'].replace({'Advanced': 0, 'Associates': 1, 'Bachelors': 2, 'High School': 3, 'Some College': 4})
# descripDf['MaritalStatus'] = descripDf['MaritalStatus'].replace({'Married': 0, 'Never Married': 1, 'Divorced': 2,  'Widowed': 3,  'Separated': 4})


# # EXPLORATORY # Count of those in group 1 vs group 0. Descriptives
# IVs: gender, severity, service, education, marital status, age, psychiatric dx
# maybe IVs: pain, confidence rating, rank
# DV: group

# In[49]:


# Compare by gender overall 
# gender = descripDf.groupby(['Group','Gender']).size().to_frame(name = 'count').reset_index()

# femaleCount = 0
# maleCount = 0
# for index,row in gender.iterrows():
#     if row['Gender'] == 'Female':
#         femaleCount += row['count']
#     else:   # only works properly if 2 genders
#         maleCount += row['count']
# percent = [(x['count']/femaleCount)*100 if x['Gender'] == 'Female' else (x['count']/maleCount)*100 for index,x in gender.iterrows()]
# gender['percent'] = percent

# display(gender)
# Compare by marital
gen0 = group0.groupby(['Group','Gender']).size().to_frame(name = 'count').reset_index()
gen0['pct'] = (gen0['count']/gen0['count'].sum())*100
display(gen0)

gen1 = group1.groupby(['Group','Gender']).size().to_frame(name = 'count').reset_index()
gen1['pct'] = (gen1['count']/gen1['count'].sum())*100
display(gen1)

gender2 = descripDf.groupby(['Gender']).size().to_frame(name = 'count').reset_index()
gender2['percent'] = (gender2, columns == ['Gender', 'count'])
gender2['percent'] = (gender2['count']/gender2['count'].sum())*100

display(gender2)


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(descripDf.Gender,descripDf.Group).plot(kind='bar')
plt.title('Gender and Evals')
plt.xlabel('Gender')
plt.ylabel('Number of patients')
plt.savefig('gender_evals')


# In[51]:


# Compare by education 
education0 = group0.groupby(['Group','Education']).size().to_frame(name = 'count').reset_index()
education0['pct'] = (education0['count']/education0['count'].sum())*100
display(education0)

education1 = group1.groupby(['Group','Education']).size().to_frame(name = 'count').reset_index()
education1['pct'] = (education1['count']/education1['count'].sum())*100
display(education1)

education = descripDf.groupby(['Education']).size().to_frame(name = 'count').reset_index()
education['pct'] = (education['count']/education['count'].sum())*100
display(education)


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(descripDf.Education,descripDf.Group).plot(kind='bar')
plt.title('Education and Evals')
plt.xlabel('Education')
plt.ylabel('Number of patients')
plt.savefig('edu_evals')


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

# Define the specific variables for the x-axis in the desired order
education_order = ['High School', 'Some College', 'Associates', 'Bachelors', 'Masters & Advanced']

# Use the specified order to sort the "Education" column
descripDf['Education'] = pd.Categorical(descripDf['Education'], categories=education_order, ordered=True)

# Sort the DataFrame based on the specified order of education levels
descripDf_sorted = descripDf.sort_values('Education')

# Create the crosstab and plot the bar chart
pd.crosstab(descripDf_sorted.Education, descripDf_sorted.Group).plot(kind='bar')

plt.title('Education and Group Comparison')
plt.xlabel('Education')
plt.ylabel('Number of patients')

# Set the x-axis labels to the predefined order of education_levels
plt.xticks(range(len(education_order)), education_order)

# Adding APA-style figure caption
caption = "Figure 2"

# Adding figure caption at the top left
plt.figtext(0.02, 0.95, caption, fontsize=8, va="top", ha="left")

plt.savefig('edu_evals')
plt.show()


# In[54]:


# Compare by severity
sev0 = group0.groupby(['Group','Severity']).size().to_frame(name = 'count').reset_index()
sev0['pct'] = (sev0['count']/sev0['count'].sum())*100
display(sev0)

sev1 = group1.groupby(['Group','Severity']).size().to_frame(name = 'count').reset_index()
sev1['pct'] = (sev1['count']/sev1['count'].sum())*100
display(sev1)

sev = descripDf.groupby(['Severity']).size().to_frame(name = 'count').reset_index()
sev['pct'] = (sev['count']/sev['count'].sum())*100
display(sev)


# In[55]:


# Compare by service
serv0 = group0.groupby(['Group','Service']).size().to_frame(name = 'count').reset_index()
serv0['pct'] = (serv0['count']/serv0['count'].sum())*100
display(serv0)

serv1 = group1.groupby(['Group','Service']).size().to_frame(name = 'count').reset_index()
serv1['pct'] = (serv1['count']/serv1['count'].sum())*100
display(serv1)

serv = descripDf.groupby(['Service']).size().to_frame(name = 'count').reset_index()
serv['pct'] = (serv['count']/serv['count'].sum())*100
display(serv)


# In[56]:


# Compare by psych
psy0 = group0.groupby(['Group','PsychiatricDX']).size().to_frame(name = 'count').reset_index()
psy0['pct'] = (psy0['count']/psy0['count'].sum())*100
display(psy0)

psy1 = group1.groupby(['Group','PsychiatricDX']).size().to_frame(name = 'count').reset_index()
psy1['pct'] = (psy1['count']/psy1['count'].sum())*100
display(psy1)

psy = descripDf.groupby(['PsychiatricDX']).size().to_frame(name = 'count').reset_index()
psy['pct'] = (psy['count']/psy['count'].sum())*100
display(psy)


# In[57]:


# Compare by marital
ms0 = group0.groupby(['Group','MaritalStatus']).size().to_frame(name = 'count').reset_index()
ms0['pct'] = (ms0['count']/ms0['count'].sum())*100
display(ms0)

ms1 = group1.groupby(['Group','MaritalStatus']).size().to_frame(name = 'count').reset_index()
ms1['pct'] = (ms1['count']/ms1['count'].sum())*100
display(ms1)

ms = descripDf.groupby(['MaritalStatus']).size().to_frame(name = 'count').reset_index()
ms['pct'] = (ms['count']/ms['count'].sum())*100
display(ms)


# In[58]:


# Compare by rank
rk0 = group0.groupby(['Group','Rank']).size().to_frame(name = 'count').reset_index()
rk0['pct'] = (rk0['count']/rk0['count'].sum())*100
display(rk0)

rk1 = group1.groupby(['Group','Rank']).size().to_frame(name = 'count').reset_index()
rk1['pct'] = (rk1['count']/rk1['count'].sum())*100
display(rk1)


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(descripDf.Rank,descripDf.Group).plot(kind='bar')
plt.title('Rank and Group Difference')
plt.xlabel('Rank')
plt.ylabel('Number of patients')
plt.savefig('rank_evals')


# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

# Define the specific variables for the x-axis in the desired order
rank_order = ['Junior Enlisted', 'Senior Enlisted', 'Officer', 'Other']

# Use the specified order to sort the "Rank" column
descripDf['Rank'] = pd.Categorical(descripDf['Rank'], categories=rank_order, ordered=True)

# Sort the DataFrame based on the specified order of ranks
descripDf_sorted_rank = descripDf.sort_values('Rank')

# Create the crosstab and plot the bar chart
pd.crosstab(descripDf_sorted_rank.Rank, descripDf_sorted_rank.Group).plot(kind='bar')

plt.title('Rank and Group Comparison')
plt.xlabel('Rank')
plt.ylabel('Number of patients')

# Set the x-axis labels to the predefined order of rank_order
plt.xticks(range(len(rank_order)), rank_order)

# Adding APA-style figure caption
caption = "Figure 1"

# Adding figure caption at the top left
plt.figtext(0.02, 0.95, caption, fontsize=8, va="top", ha="left")

plt.savefig('rank_evals')
plt.show()


# In[ ]:





# In[61]:


# Compare by confidence
cr0 = group0.groupby(['Group','ConfidenceRating']).size().to_frame(name = 'count').reset_index()
cr0['pct'] = (cr0['count']/cr0['count'].sum())*100
display(cr0)

cr1 = group1.groupby(['Group','ConfidenceRating']).size().to_frame(name = 'count').reset_index()
cr1['pct'] = (cr1['count']/cr1['count'].sum())*100
display(cr1)


# In[62]:


descripDf.groupby('Group').mean()


# In[63]:


descripDf.groupby('Service').mean()


# In[64]:


descripDf.groupby('Gender').mean()


# In[65]:


descripDf.groupby('Education').mean()


# ## STAT MODELS ##

# In[66]:


list(descripDf.columns)


# In[67]:


# REDUCE FURTHER FOR DUMMY CODING
to_drop2 = ['pseudo_id',
 'NumberOfEvals',
 'Location',
 'ReferralSource',
 'Status',
 'TBI_DX',
 'PrimaryDX',
 'Comorbid_PSYC_DX_Other_Psych_dx',
 'ConfidenceRating',
 'PainScale',
 'Evaluation',
 'ProgramUsed']

descripDf.drop(to_drop2, inplace=True, axis=1)


# In[68]:


list(descripDf.columns)


# In[69]:


descripDf.describe()


# In[70]:


# Test log reg 
import statsmodels.formula.api as smf
import patsy
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# ## BELOW IS OLD FROM 1/26-1/27 USING STAT MODELS ##

# In[71]:


sns.regplot(x='PatientAge', y='Group', y_jitter = 0.03, data = descripDf, logistic = True, ci=None)
plt.show()


# In[72]:


# Test for collinearity
# y, X = dmatrices('Group ~ Gender + Education + MaritalStatus + PsychiatricDX + Rank + Severity + Service + PatientAge', data=descripDf, return_type='dataframe')
y, X = dmatrices('Group ~ C(Gender, Treatment(reference="Male")) + C(Education, Treatment(reference="Some College")) + C(MaritalStatus, Treatment(reference = "Married")) + PsychiatricDX +  C(Severity, Treatment(reference ="Mild")) + C(Service, Treatment(reference = "USA")) + PatientAge' , data=descripDf, return_type='dataframe')

#create DataFrame to hold VIF values
vif_df = pd.DataFrame()
vif_df['variable'] = X.columns 

#calculate VIF for each predictor variable 
vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#view VIF for each predictor variable 
display(vif_df)

# VIF = 1: There is no correlation between a given predictor variable and any other predictor variables in the model.
# VIF between 1 and 5: There is moderate correlation between a given predictor variable and other predictor variables in the model.


# In[73]:


model = smf.logit('Group ~ C(Gender, Treatment(reference="Male")) +   C(Rank, Treatment(reference = "Junior Enlisted")) + C(Education, Treatment(reference="Some College")) + C(MaritalStatus, Treatment(reference = "Married")) + PsychiatricDX +  C(Severity, Treatment(reference ="Mild")) + C(Service, Treatment(reference = "USA")) + PatientAge',data=descripDf).fit()
display(model.summary())

# As such, the reference category should be the category that allows for easier interpretation and the one that you care about the most. Mathematically speaking, this could be the category that has the largest representation in your data, i.e. that which represents the ‘norm’. Equally, it could be the category that has an unexpected presence in your data.


# In[74]:


display(model.get_margeff(at ='overall').summary())


# In[75]:


model2 = smf.logit('Group ~ C(Gender, Treatment(reference="Male")) + C(Education, Treatment(reference="Some College")) + C(MaritalStatus, Treatment(reference = "Married")) + C(Severity, Treatment(reference ="Mild")) + C(Service, Treatment(reference = "USA"))  + PsychiatricDX +   PatientAge', data=descripDf).fit()

# DO NOT INCLUDE RANK AND REMOVE OTHERS GROUP FROM RANK... LEADS TO INSIG. 
#model2 = smf.logit('Group ~ C(Gender, Treatment(reference="Male")) + C(Rank, Treatment(reference="Junior Enlisted")) + PatientAge', data=descripDf).fit()
display(model2.summary())


# In[76]:


import pandas as pd

# Assuming you have the DataFrame coefs2
coefs2 = pd.DataFrame({
    'coef': model2.params.values,
    'odds ratio': np.exp(model2.params.values),
    'name': model2.params.index
})

# Set display option to show full column width
pd.set_option('display.max_colwidth', None)

# Display the expanded table
display(coefs2)


# In[77]:


display(model2.get_margeff(at ='overall').summary())


# **Descriptives for **

# In[78]:


# Compare by gender and primary dx 
# evalPrimary = droppedDup.groupby(['group','PrimaryDX']).size().to_frame(name = 'count').reset_index()
# display(evalPrimary)


# In[79]:


# Compare by gender and primary dx 
# gen = droppedDup.groupby(['group', 'gender']).size().to_frame(name = 'count').reset_index()
# display(gen)


# In[80]:


# See count for referral source to check IOP and 4-weekers
# tester = droppedDup.groupby(['group', 'ReferralSource']).size().to_frame(name = 'count').reset_index()
# display(tester)


# In[81]:


params = model2.params
conf = model2.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
# convert log odds to ORs
odds = pd.DataFrame(np.exp(conf))
# check if pvalues are significant
odds['pvalues'] = model2.pvalues
odds['significant?'] = ['significant' if pval <= 0.05 else 'not significant' for pval in model2.pvalues]
odds


# In[82]:


fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(10, 6), dpi=150)

# Define custom row names
row_names = [
    'Gender (Female)', 'Education (High School)','Education (Associates)', 'Education (Bachelors)',
    'Education (Masters and Advanced)', 'Marital Status (Never Married)', 'Marital Status (Other)',
    'Severity (Moderate)', 'Severity (Penetrating)', 'Severity (Severe)', 'Service (USAF)', 'Service (USMC)',
    'Service (USN)', 'PsychiatricDX', 'Patient Age', 'Intercept'
]

# Define the predictor variable names and their corresponding odds ratios
predictors = [
    'C(Gender, Treatment(reference="Male"))[T.Female]', 'C(Education, Treatment(reference="Some College"))[T.High School]', 'C(Education, Treatment(reference="Some College"))[T.Associates]', 'C(Education, Treatment(reference="Some College"))[T.Bachelors]',
    'C(Education, Treatment(reference="Some College"))[T.Masters & Advanced]', 'C(MaritalStatus, Treatment(reference="Married"))[T.Never Married]', 'C(MaritalStatus, Treatment(reference="Married"))[T.Other]',
    'C(Severity, Treatment(reference="Mild"))[T.Moderate]', 'C(Severity, Treatment(reference="Mild"))[T.Penetrating]', 'C(Severity, Treatment(reference="Mild"))[T.Severe]', 'C(Service, Treatment(reference="USA"))[T.USAF]', 'C(Service, Treatment(reference="USA"))[T.USMC]',
    'C(Service, Treatment(reference="USA"))[T.USN]', 'PsychiatricDX', 'PatientAge', 'Intercept'
]

for idx, row in odds.iloc[::-1].iterrows():
    ci = [[row['Odds Ratio'] - row[::-1]['2.5%']], [row['97.5%'] - row['Odds Ratio']]]
    if row['significant?'] == 'significant':
        plt.errorbar(x=[row['Odds Ratio']], y=[row_names[predictors.index(idx)]], xerr=ci,
            ecolor='tab:red', capsize=3, linestyle='None', linewidth=1, marker="o", 
                     markersize=5, mfc="tab:red", mec="tab:red")
    else:
        plt.errorbar(x=[row['Odds Ratio']], y=[row_names[predictors.index(idx)]], xerr=ci,
            ecolor='tab:gray', capsize=3, linestyle='None', linewidth=1, marker="o", 
                     markersize=5, mfc="tab:gray", mec="tab:gray")

plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.ylabel('Predictor Variables')
plt.tight_layout()

# Add the legend
plt.legend(handles=[plt.Line2D([], [], color='tab:red', marker='o', markersize=5, linestyle='None', label='Significant'),
                    plt.Line2D([], [], color='tab:gray', marker='o', markersize=5, linestyle='None', label='Not Significant')],
           loc='upper right', fontsize=8)

plt.savefig('forest_plot.png')
plt.show()


# In[83]:


import matplotlib.pyplot as plt

predictors = [
    'Gender', 'Education (Associates)', 'Education (Bachelors)', 'Education (High School)',
    'Education (Masters & Advanced)', 'Marital Status (Never Married)', 'Marital Status (Other)',
    'Severity (Moderate)', 'Severity (Penetrating)', 'Severity (Severe)', 'Service (USAF)', 'Service (USMC)',
    'Service (USN)', 'PsychiatricDX', 'PatientAge'
]

coefficients = [-0.0980, 0.2612, 0.0817, -0.5674, 0.5513, 0.2478, 0.0125, -0.2514, 0.9409, 0.4156,
                0.3597, -0.3855, -0.0761, -0.1981, 0.0043]

# Set the figure size and create a bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(predictors, coefficients, color='lightgrey', alpha=0.8)

# Highlight the "Education (High School)" row in red
highlight_index = predictors.index('Education (High School)')
bars[highlight_index].set_color('red')

# Add labels and title
plt.xlabel('Coefficient')
plt.ylabel('Predictor Variables')

# Display grid lines
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




