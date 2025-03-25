import pandas as pd
import ast
from process_data import CategoryTransformer
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == '__main__':
    df = pd.read_csv('data/raw/full_data_2020_2025_FORD.csv')

    category_tranformer =  CategoryTransformer()
    
    df[['problem_type', 'problem_cause']] = df['summary'].apply(lambda x: category_tranformer.extract_pieces_and_problems(x) if isinstance(x, str) else ('undefined', 'undefined', 'undefined')).apply(pd.Series)
    df = category_tranformer.transform_categories(df, is_training=True)
    df[['summary', 'components', 'problem_type', 'problem_cause']].to_csv('./data/processed/df_categories.csv',index=False)

    # Showing the distribution of components
    components_count_plt = df['components'].explode().value_counts()
    components_count = components_count_plt.reset_index(name='count')
    components_count.columns = ['Components', 'Count']
    print(components_count)

    # Showing the distribution of problem_type
    problem_count_plt = df['problem_type'].explode().value_counts()
    problem_count = problem_count_plt.reset_index(name='count')
    problem_count.columns = ['Problem Type', 'Count']
    print(problem_count)

    # Showing the distribution of problem_cause
    cause_count_plt = df['problem_cause'].explode().value_counts()
    cause_count = cause_count_plt.reset_index(name='count')
    cause_count.columns = ['Problem Cause', 'Count']
    print(cause_count)


    # Plotting the distribution charts
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  

    # Plot the first chart (larger one) to span the full first row
    ax1 = fig.add_subplot(gs[0, :]) 
    x = range(len(components_count['Components']))
    bars1 = ax1.bar(x, components_count['Count'], color='cyan')
    ax1.set_title('Pieces Count')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Instances')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components_count['Components'], rotation=45, ha='right')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # Plot the second chart in the second row, first column
    ax2 = fig.add_subplot(gs[1, 0])
    x2 = range(len(cause_count['Problem Cause']))
    bars2 = ax2.bar(x2, cause_count['Count'], color='olive')
    ax2.set_title('Cause Count')
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Number of Instances')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cause_count['Problem Cause'], rotation=45, ha='right')

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# Plot the third chart in the second row, second column
    ax3 = fig.add_subplot(gs[1, 1])
    x3 = range(len(problem_count['Problem Type']))
    bars3 = ax3.bar(x3, problem_count['Count'], color='brown')
    ax3.set_title('Problem Count')
    ax3.set_xlabel('Categories')
    ax3.set_ylabel('Number of Instances')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(problem_count['Problem Type'], rotation=45, ha='right')

    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('data/outputs/categories_distribution.png')
    plt.show()

