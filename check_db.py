import sqlite3

try:
    conn = sqlite3.connect('ey_navigator.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables found:", [t[0] for t in tables])
    
    print("\n--- Predictions ---")
    cursor.execute("SELECT id, target_role, feasibility_score FROM career_predictions ORDER BY id DESC LIMIT 5;")
    preds = cursor.fetchall()
    for p in preds:
        print(p)

    print("\n--- Ethics Audits ---")
    cursor.execute("SELECT * FROM ethics_audits ORDER BY id DESC LIMIT 5;")
    audits = cursor.fetchall()
    for a in audits:
        print(a)
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
