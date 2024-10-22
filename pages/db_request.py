import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv
DATABASE_CONFIG = {
    "dbname": "FDX_AI",
    "user": "postgres",
    "password": "Trinhchau310104",
    "host": "localhost",
    "port": 5432
}


@contextmanager
def db_connection():
    """Context manager for handling database connection and cursor."""
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        yield cur
        conn.commit()
    except Exception as error:
        conn.rollback()
        print(f"Error: {error}")
    finally:
        cur.close()
        conn.close()


# Example usage: Fetch data
def get_messages(username, conversation_id):
    query = f"""select * from  users  u left join conversations con on 
            u.user_id = con.user_id
            left join messages me on con.conversation_id = me.conversation_id

            where username = '{username}' and con.conversation_id = '{conversation_id}'
            """
    with db_connection() as cur:
        cur.execute(query)
        result = cur.fetchall()
        return result


def add_messages(conversation_id, role, msg):
    query = f"""INSERT INTO messages (conversation_id, sender, message_text)
                VALUES 
                ({conversation_id}, '{role}', '{msg}');
                """
    with db_connection() as cur:
        cur.execute(query)
    # Example usage: Insert data


def add_user(username, password):
    query = "INSERT INTO users (username, password) VALUES (%s, %s)"
    with db_connection() as cur:
        cur.execute(query, (username, password))


def get_latest_conversation(username):
    query = f"""
            select MAX(con.conversation_id) from  users  u left join conversations con on 
            u.user_id = con.user_id
            left join messages me on con.conversation_id = me.conversation_id
            where username = '{username}'
;"""
    with db_connection() as cur:
        cur.execute(query)
        result = cur.fetchall()
        return result[0][0]


def add_conversation(user_name):

    query = f"""INSERT INTO conversations (conversation_name, user_id)
        VALUES 
        ('New conversation', (select user_id from    users where username  = '{user_name}'));"""
    with db_connection() as cur:
        cur.execute(query)


def get_conversation_name(conversation_id):
    query = f"""
                select conversation_name   from conversations
where conversation_id= '{conversation_id}'
    ;"""
    with db_connection() as cur:
        cur.execute(query)
        result = cur.fetchall()
        return result[0][0]
def get_conversation_list(user_name):


    query = f"""select * from  users  u left join conversations con on 
                    u.user_id = con.user_id
                    where username = '{user_name}'
                    """
    with db_connection() as cur:
        cur.execute(query)
        result = cur.fetchall()
        return result
def update_conversation_name(conversation_id, new_name):
    query = f"""
        UPDATE conversations
        SET conversation_name = '{new_name}'
        WHERE conversation_id = '{conversation_id}';
    """
    with db_connection() as cur:
        cur.execute(query)

# Example usage: Update data
def update_user_password(user_id, new_password):
    query = "UPDATE users SET password = %s WHERE user_id = %s"
    with db_connection() as cur:
        cur.execute(query, (new_password, user_id))


def check_login(username, password):
    query = f"""
        SELECT 
        CASE  
            WHEN EXISTS (SELECT 1 FROM users WHERE username = '{username}'AND password = '{password}' ) THEN TRUE
        ELSE FALSE
    END AS user_exists;"""
    with db_connection() as cur:
        cur.execute(query)
        result = cur.fetchall()
        return result[0][0]


# Example usage: Delete user
def delete_user(user_id):
    query = "DELETE FROM users WHERE user_id = %s"
    with db_connection() as cur:
        cur.execute(query, (user_id,))

# login('john_doe','password123')
# print(check_login('john_doe','password123'))

# print(get_messages('john_doe','1'))
# print(get_conversation_name('1'))
