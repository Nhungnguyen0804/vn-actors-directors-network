import pandas as pd
import random
import json
import re

# Load datasets
try:
    df_acted = pd.read_csv('vietnamese_cinema_acted_in.csv')
    df_collab = pd.read_csv('vietnamese_cinema_collaborations.csv')
    df_directed = pd.read_csv('vietnamese_cinema_directed.csv')
    df_films = pd.read_csv('vietnamese_cinema_films.csv')
    df_persons = pd.read_csv('vietnamese_cinema_persons.csv')
except Exception as e:
    print(f"Error loading files: {e}")

# --- Helpers ---
def clean_name(text):
    if pd.isna(text): return None
    # Remove things like (1990), (diễn viên), etc if needed, or keep them if they are identifiers.
    # The snippet shows "Trấn Thành" is clean, but spouse has years "Hari Won (2016)".
    text = str(text)
    match = re.match(r"([^(]+)", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def clean_film_title(text):
    if pd.isna(text): return None
    return str(text).strip()

# Prepare Data Structures
df_persons['clean_spouse'] = df_persons['spouse'].apply(clean_name)
person_spouse_map = df_persons.dropna(subset=['clean_spouse']).set_index('name')['clean_spouse'].to_dict()
all_spouses = list(set(person_spouse_map.values()))

film_director_map = df_films.dropna(subset=['director']).set_index('title')['director'].to_dict()
all_directors = list(set(film_director_map.values()))

film_genre_map = df_films.dropna(subset=['genre']).set_index('title')['genre'].to_dict()
all_genres = []
for g in film_genre_map.values():
    if pd.isna(g): continue
    parts = [x.strip() for x in str(g).replace('*', ',').split(',')]
    all_genres.extend(parts)
all_genres = list(set(all_genres))

# Actor -> Films and Film -> Actors
# Use df_acted
# Clean film titles in acted to match df_films if needed, but assuming consistency for now
actor_films = df_acted.groupby('person_name')['film_title'].apply(list).to_dict()
film_actors = df_acted.groupby('film_title')['person_name'].apply(list).to_dict()
# Also map (film, actor) -> character
film_actor_char = df_acted.set_index(['film_title', 'person_name'])['character'].to_dict()

all_films = list(film_actors.keys())
all_actors = list(actor_films.keys())

# --- Generation Functions ---

dataset = []
target_count = 2000

# Helper to create question object
def create_q_obj(question_text, correct_ans, distractors, reasoning):
    options = distractors + [correct_ans]
    random.shuffle(options)
    
    labels = ["A", "B", "C", "D"]
    formatted_options = []
    correct_label = ""
    
    for i, opt in enumerate(options):
        formatted_options.append(f"{labels[i]}. {opt}")
        if opt == correct_ans:
            correct_label = labels[i]
            
    return {
        "question": question_text,
        "type": "multi-hop",
        "options": formatted_options,
        "answer": correct_label,
        "reasoning": reasoning
    }

# 1. Actor Spouse via Film (Film -> Actor -> Spouse)
# "Vợ/chồng của diễn viên đóng vai [Char] trong phim [Film] là ai?"
candidates_1 = []
for actor, spouse in person_spouse_map.items():
    if actor in actor_films:
        for film in actor_films[actor]:
            char = film_actor_char.get((film, actor))
            if pd.isna(char):
                q_text = f"Vợ/chồng của diễn viên {actor} trong phim '{film}' là ai?"
            else:
                q_text = f"Vợ/chồng của diễn viên đóng vai '{char}' trong phim '{film}' là ai?"
            
            candidates_1.append((q_text, spouse, f"{film} -> {actor} -> {spouse}"))

# 2. Common Films (Actor A + Actor B -> Film)
candidates_2 = []
# Use collaborations df
# Filter for meaningful collaborations (count >= 1)
for idx, row in df_collab.iterrows():
    if row['film_count'] >= 1 and pd.notna(row['films']):
        a = row['person1']
        b = row['person2']
        films = str(row['films']).split('|')
        for f in films:
            q_text = f"Phim nào có sự tham gia của cả {a} và {b}?"
            candidates_2.append((q_text, f, f"{a}, {b} -> {f}"))

# 3. Director of Film with Actor (Actor -> Film -> Director)
candidates_3 = []
for film, director in film_director_map.items():
    if film in film_actors:
        # Pick all actors to generate variations
        for actor in film_actors[film]:
            q_text = f"Đạo diễn của bộ phim '{film}' có sự tham gia của {actor} là ai?"
            candidates_3.append((q_text, director, f"{actor} -> {film} -> {director}"))

# 4. Genre of Film with Actor (Actor -> Film -> Genre)
candidates_4 = []
for film, genre in film_genre_map.items():
    if film in film_actors:
        for actor in film_actors[film]:
            q_text = f"Thể loại của bộ phim '{film}' do {actor} đóng là gì?"
            candidates_4.append((q_text, genre, f"{actor} -> {film} -> {genre}"))

# 5. Spouse of Director (Film -> Director -> Spouse)
candidates_5 = []
# Need to link director name to person name in person_spouse_map
# Assuming names match
for film, director in film_director_map.items():
    # Director might be a list or single name. Simple string match for now.
    # Check if director is in person_spouse_map
    # Sometimes director field has multiple names.
    dir_names = [d.strip() for d in str(director).replace('*', ',').split(',')]
    for d_name in dir_names:
        if d_name in person_spouse_map:
            spouse = person_spouse_map[d_name]
            q_text = f"Vợ/chồng của đạo diễn bộ phim '{film}' ({d_name}) là ai?"
            candidates_5.append((q_text, spouse, f"{film} -> {d_name} -> {spouse}"))

# --- Sampling to reach 2000 ---
# Prioritize variety.
# We have lists of (question, answer, reasoning). We need to generate distractors on the fly.

all_candidates = []

# Add type tag to handle distractors
for c in candidates_1: all_candidates.append(list(c) + ['spouse'])
for c in candidates_2: all_candidates.append(list(c) + ['film'])
for c in candidates_3: all_candidates.append(list(c) + ['director'])
for c in candidates_4: all_candidates.append(list(c) + ['genre'])
for c in candidates_5: all_candidates.append(list(c) + ['spouse'])

random.shuffle(all_candidates)

# De-duplicate by question text
unique_q_map = {}
for item in all_candidates:
    q_text = item[0]
    if q_text not in unique_q_map:
        unique_q_map[q_text] = item

final_candidates = list(unique_q_map.values())
if len(final_candidates) > target_count:
    final_candidates = final_candidates[:target_count]

# Generate JSON
output_json = []

for item in final_candidates:
    q_text, correct, reasoning, q_type = item
    
    distractors = []
    
    if q_type == 'spouse':
        pool = [s for s in all_spouses if s != correct]
        if len(pool) < 3: pool = all_spouses # Fallback
        distractors = random.sample(pool, min(3, len(pool)))
        
    elif q_type == 'film':
        pool = [f for f in all_films if f != correct]
        distractors = random.sample(pool, min(3, len(pool)))
        
    elif q_type == 'director':
        pool = [d for d in all_directors if d != correct]
        distractors = random.sample(pool, min(3, len(pool)))
        
    elif q_type == 'genre':
        pool = [g for g in all_genres if g != correct]
        distractors = random.sample(pool, min(3, len(pool)))
    
    # Pad distractors if not enough (rare case)
    while len(distractors) < 3:
        distractors.append("Không xác định")
    
    q_obj = create_q_obj(q_text, correct, distractors, reasoning)
    output_json.append(q_obj)

# Save to file
with open('evaluation_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"Generated {len(output_json)} questions.")