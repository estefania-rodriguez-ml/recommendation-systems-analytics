import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import ast
import json
import zipfile
import pickle

df_dashboard = pd.read_csv("df_dashboard.csv")
df_dashboard['tag_name'] = df_dashboard['tag_name'].apply(ast.literal_eval)

df_dashboard = df_dashboard.rename(columns={
    'title': 'Book Title',
    'tag_name': 'Genre',
    'ratings_count': 'Number of Ratings',
    'average_rating': 'Average Rating',
    'rating': 'User Rating',
    'score': 'Weighted Score',
    'book_id': 'Book ID'
})

useful_genres = ['fiction', 'classic', 'kids', 'children-s-lit', 
                 'bio-memoir', 'war', 'history', 'romance', 
                 'mystery', 'fantasy']


st.title("A book recommendation system for readers aged 65+ based on the preferences of Goodreads users")

tab1, tab2, tab3 = st.tabs([
    "The Top General Reads Books by Genre",
    "The Top 10 Most Read Books for Adults, Personalised by Average Rating",
    "Recommendations"
])



with tab1:
    st.header("Most Read Books by Genre")
    st.write("""
        The most frequently read books from the Goodreads community sample,
        organised by genre.
    """)

    df_exploded = df_dashboard.explode('Genre')

    adult_genres = ['Fiction', 'Classic', 'Bio-memoir', 'War', 'History',
                    'Romance', 'Mystery', 'Fantasy']

    book_counts = (
        df_exploded[df_exploded['Genre'].isin(adult_genres)]
        .groupby(['Genre', 'Book Title'])
        .size()
        .reset_index(name='Count')
        .sort_values(['Genre', 'Count'], ascending=[True, False])
    )

    selected = []
    rows = []
    for genre in book_counts['Genre'].unique():
        candidates = book_counts[book_counts['Genre'] == genre]
        for _, r in candidates.iterrows():
            if r['Book Title'] not in selected:
                selected.append(r['Book Title'])
                rows.append(r)
                break

    top_per_genre_distinct = pd.DataFrame(rows).sort_values('Count', ascending=False)

    fig_books = px.bar(
        top_per_genre_distinct,
        x='Genre',
        y='Count',
        color='Genre',
        color_discrete_sequence=px.colors.sequential.Plasma,
        title='The Top General Reads Books by Genre',
        hover_data={'Book Title': True}
    )

    fig_books.update_layout(
        title_x=0.5,
        xaxis_title='Genre',
        yaxis_title='Number of Reads',
        showlegend=False
    )

    st.dataframe(top_per_genre_distinct)
    st.plotly_chart(fig_books)



top_books_filtered = pd.read_csv("top_books_filtered.csv")

top_books_filtered = top_books_filtered.rename(columns={
    'title': 'Book Title',
    'tag_name': 'Genre',
    'ratings_count': 'Number of Ratings',
    'average_rating': 'Average Rating',
    'rating': 'User Rating',
    'score': 'Weighted Score'
})

with tab2:

    st.header("Top 10 Most Read Adult Books (Weighted Ranking Score)")
    st.write("""
    These books are ranked using a weighted scoring formula that balances:
    - **Average rating**
    - **Number of ratings**
    """)
    st.info("""
    You can use this chart to compare books across genres.  
    If you are interested in a particular genre, look for its associated colour.
    """)

    
    colors_full = px.colors.sequential.Plasma
    
    color_indices = [int(i * (len(colors_full)-1)/9) for i in range(10)]
    colors_for_books = [colors_full[i] for i in color_indices]

    fig_top_books = px.bar(
        top_books_filtered,
        y='Book Title',
        x='Weighted Score',
        orientation='h',
        color='Genre',
        title="The Top 10 Most Read Books for Adults, Personalised by Average Rating",
        color_discrete_sequence=colors_for_books
    )


    fig_top_books.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Weighted Score",
        yaxis_title="Book Title",
        title={
        "text": "The Top 10 Most Read Books for Adults, Personalised by Average Rating",
        "x": 0.5,
        "xanchor": "center"},
        showlegend=True
)


    st.dataframe(top_books_filtered)
    st.plotly_chart(fig_top_books)

with tab3:
    
    st.header("Item-Item Recommendations")
    st.write("""
    These recommendations are based on what other readers with similar interests have read. 
    Books suggested here are often enjoyed by people who read the book you select.
    """)

    item_similarity_df = pd.read_pickle("item_similarity_df_small.pkl")

    with open("top_30_similar_items.pkl", "rb") as f:
        top_30_similar_items = pickle.load(f)
    with open("book_titles.pkl", "rb") as f:
        book_titles = pickle.load(f)

    valid_ids = set(top_30_similar_items.keys())
    book_options = {v: k for k, v in book_titles.items() if k in valid_ids}

    selected_book_title = st.selectbox(
        "Select a book you enjoyed (Item-Item):",
        sorted(book_options.keys())
    )
    selected_book_id = book_options[selected_book_title]
    similar_ids = top_30_similar_items[selected_book_id][:10]

    df_reco = pd.DataFrame({
        "Recommended Book": [book_titles[i] for i in similar_ids],
        "Similarity Score": [
            item_similarity_df.loc[selected_book_id, i] 
            if i in item_similarity_df.columns else 0
            for i in similar_ids
        ]
    })
    df_reco['Recommended Book Clean'] = df_reco['Recommended Book'].str.replace(r'^(A |An |The )', '', regex=True)
    df_reco_sorted = df_reco.sort_values(by='Recommended Book Clean')

    st.subheader("Item-Item Recommendations:")
    st.dataframe(
        df_reco_sorted[['Recommended Book Clean']].rename(
            columns={'Recommended Book Clean': 'Recommended Book'}
        )
    )

    
    st.header("Content-Based Recommendations")
    st.write("""
    These recommendations are based on the content of each book—such as genres, themes, and tags assigned by other readers.  
    You can explore books similar in style or subject to the ones you select.  
    """)

    
    try:
        df_sample = pd.read_csv("df_sample.csv")
        with open("recommendations_sample.json", "r") as f:
            recommendations = json.load(f)
    except Exception as e:
        st.error(f"Error loading content-based recommendations: {e}")
        df_sample = pd.DataFrame()
        recommendations = {}

    if not df_sample.empty:
        book_choice_cb = st.selectbox("Select a book (Content-Based):", df_sample['title'])
        if book_choice_cb in recommendations:
            df_content_reco = pd.DataFrame({
                "Recommended Book": recommendations[book_choice_cb]
            })
            df_content_reco['Recommended Book Clean'] = df_content_reco['Recommended Book'].str.replace(r'^(A |An |The )', '', regex=True)
            df_content_reco_sorted = df_content_reco.sort_values(by='Recommended Book Clean')
            st.subheader("Content-Based Recommendations:")
            st.dataframe(
                df_content_reco_sorted[['Recommended Book Clean']].rename(
                    columns={'Recommended Book Clean': 'Recommended Book'}
                )
            )
        else:
            st.write("No recommendations available for this selection.")

    
     
    st.header("Top Authors and Their Books")
    st.write("""
    Explore authors and the books they have written.
    These recommendations are based on the top-performing authors. The top 10 authors on this list have received high average ratings from other readers.      
    """)

    books_by_author = (
        df_dashboard.groupby('authors')['Book Title']
        .apply(lambda x: ', '.join(sorted(x.unique())))
        .reset_index()
    )

    books_per_author = df_dashboard.groupby('authors')['Book ID'].nunique().reset_index()
    books_per_author.rename(columns={'Book ID':'Number of Books'}, inplace=True)

    authors_df = books_by_author.merge(books_per_author, on='authors')
    authors_df = authors_df.sort_values('Number of Books', ascending=False).head(10)

    
    authors_df = authors_df.rename(columns={'authors': 'Author'})

    st.dataframe(
        authors_df[['Author', 'Number of Books', 'Book Title']]
    )

    for idx, row in authors_df.iterrows():
        with st.expander(f"{row['Author']} ({row['Number of Books']} books)"):
            st.write(row['Book Title'])