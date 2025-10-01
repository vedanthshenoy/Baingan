import mysql.connector
from mysql.connector import Error
import hashlib
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, host='localhost', database='baingan_db', user='root', password=''):
        """Initialize database connection parameters."""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
    
    def connect(self):
        """Create database connection."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            cursor = self.connection.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(64) NOT NULL,
                    salt VARCHAR(32) NOT NULL,
                    user_type ENUM('registered', 'guest') DEFAULT 'registered',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL
                )
            """)
            
            # Export results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS export_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_name VARCHAR(100) NOT NULL,
                    unique_id VARCHAR(255) UNIQUE NOT NULL,
                    test_type VARCHAR(50),
                    prompt_name VARCHAR(255),
                    system_prompt TEXT,
                    query TEXT,
                    response LONGTEXT,
                    status VARCHAR(50),
                    status_code VARCHAR(20),
                    timestamp VARCHAR(50),
                    edited BOOLEAN DEFAULT FALSE,
                    step VARCHAR(100),
                    combination_strategy VARCHAR(100),
                    combination_temperature VARCHAR(20),
                    slider_weights TEXT,
                    rating INT,
                    remark TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user_name (user_name),
                    INDEX idx_test_type (test_type),
                    INDEX idx_rating (rating),
                    INDEX idx_timestamp (timestamp)
                )
            """)
            
            # Shared datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_datasets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    created_by VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_created_by (created_by)
                )
            """)
            
            # Shared dataset items table (many-to-many relationship)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_dataset_items (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    dataset_id INT NOT NULL,
                    export_unique_id VARCHAR(255) NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES shared_datasets(id) ON DELETE CASCADE,
                    FOREIGN KEY (export_unique_id) REFERENCES export_results(unique_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_dataset_item (dataset_id, export_unique_id)
                )
            """)
            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error creating tables: {e}")
            return False
    
    def hash_password(self, password, salt=None):
        """Hash password with salt using SHA-256."""
        if salt is None:
            salt = os.urandom(16).hex()
        
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return pwd_hash, salt
    
    def register_user(self, username, password, user_type='registered'):
        """Register a new user."""
        try:
            cursor = self.connection.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                cursor.close()
                return False, "Username already exists"
            
            # Hash password
            pwd_hash, salt = self.hash_password(password)
            
            # Insert new user
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, user_type)
                VALUES (%s, %s, %s, %s)
            """, (username, pwd_hash, salt, user_type))
            
            self.connection.commit()
            cursor.close()
            return True, "User registered successfully"
        except Error as e:
            print(f"Error registering user: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate a user with username and password."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get user details
            cursor.execute("""
                SELECT id, username, password_hash, salt, user_type
                FROM users
                WHERE username = %s
            """, (username,))
            
            user = cursor.fetchone()
            
            if not user:
                cursor.close()
                return False, None, "User not found"
            
            # Verify password
            pwd_hash, _ = self.hash_password(password, user['salt'])
            
            if pwd_hash == user['password_hash']:
                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = %s WHERE id = %s
                """, (datetime.now(), user['id']))
                self.connection.commit()
                cursor.close()
                return True, user, "Login successful"
            else:
                cursor.close()
                return False, None, "Invalid password"
        except Error as e:
            print(f"Error authenticating user: {e}")
            return False, None, f"Authentication failed: {str(e)}"
    
    def create_guest_session(self, guest_name):
        """Create a guest session."""
        try:
            cursor = self.connection.cursor()
            
            # Create a unique guest username
            guest_username = f"guest_{guest_name}_{os.urandom(4).hex()}"
            
            # Generate a random password for guest
            guest_password = os.urandom(16).hex()
            pwd_hash, salt = self.hash_password(guest_password)
            
            # Insert guest user
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, user_type)
                VALUES (%s, %s, %s, 'guest')
            """, (guest_username, pwd_hash, salt))
            
            self.connection.commit()
            user_id = cursor.lastrowid
            cursor.close()
            
            return True, {
                'id': user_id,
                'username': guest_username,
                'display_name': guest_name,
                'user_type': 'guest'
            }
        except Error as e:
            print(f"Error creating guest session: {e}")
            return False, None
    
    def get_user_by_id(self, user_id):
        """Get user information by ID."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, username, user_type, created_at, last_login
                FROM users
                WHERE id = %s
            """, (user_id,))
            
            user = cursor.fetchone()
            cursor.close()
            return user
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
        
    def check_user_type(self, username):
        """Check the user_type for a given username in the users table."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT user_type FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            cursor.close()
            return result['user_type'] if result else None
        except Error as e:
            print(f"Error checking user type: {e}")
            return None
    
    def delete_guest_users(self, days_old=7):
        """Delete guest users older than specified days."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM users
                WHERE user_type = 'guest'
                AND created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
            """, (days_old,))
            
            self.connection.commit()
            deleted_count = cursor.rowcount
            cursor.close()
            return True, deleted_count
        except Error as e:
            print(f"Error deleting guest users: {e}")
            return False, 0
    
    def save_export_result(self, user_name, unique_id, test_type, prompt_name, system_prompt, 
                          query, response, status, status_code, timestamp, edited, step,
                          combination_strategy, combination_temperature, slider_weights, 
                          rating, remark, created_at=None, updated_at=None):
        """Save export result to database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO export_results (
                    user_name, unique_id, test_type, prompt_name, system_prompt, query, 
                    response, status, status_code, timestamp, edited, step,
                    combination_strategy, combination_temperature, slider_weights, 
                    rating, remark
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    response = VALUES(response),
                    status = VALUES(status),
                    status_code = VALUES(status_code),
                    edited = VALUES(edited),
                    rating = VALUES(rating),
                    remark = VALUES(remark)
            """, (user_name, unique_id, test_type, prompt_name, system_prompt, query,
                  response, status, status_code, timestamp, edited, step,
                  combination_strategy, combination_temperature, slider_weights,
                  rating, remark))
            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error saving export result: {e}")
            return False
    
    def get_user_export_results(self, user_name):
        """Get all export results for a specific user."""
        try:
            import pandas as pd
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM export_results
                WHERE user_name = %s
                ORDER BY timestamp DESC
            """, (user_name,))
            
            results = cursor.fetchall()
            cursor.close()
            
            if results:
                return pd.DataFrame(results)
            return pd.DataFrame()
        except Error as e:
            print(f"Error fetching export results: {e}")
            return pd.DataFrame()
    
    def update_export_rating(self, unique_id, rating, remark):
        """Update rating and remark for a specific export result."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE export_results
                SET rating = %s,
                    remark = %s,
                    edited = TRUE,
                    updated_at = %s
                WHERE unique_id = %s
            """, (rating, remark, datetime.now(), unique_id))

            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error updating export rating: {e}")
            return False
    
    def delete_export_result(self, unique_id):
        """Delete a specific export result."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM export_results
                WHERE unique_id = %s
            """, (unique_id,))
            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error deleting export result: {e}")
            return False
    
    def delete_all_user_export_results(self, user_name):
        """Delete all export results for a specific user."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM export_results
                WHERE user_name = %s
            """, (user_name,))
            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error deleting all user export results: {e}")
            return False
    
    def get_export_statistics(self, user_name):
        """Get statistics about user's export results."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_results,
                    COUNT(CASE WHEN rating IS NOT NULL THEN 1 END) as rated_results,
                    AVG(rating) as avg_rating,
                    COUNT(DISTINCT test_type) as unique_test_types,
                    MIN(timestamp) as first_result,
                    MAX(timestamp) as last_result
                FROM export_results
                WHERE user_name = %s
            """, (user_name,))
            
            stats = cursor.fetchone()
            cursor.close()
            return stats
        except Error as e:
            print(f"Error fetching export statistics: {e}")
            return None
    
    def create_shared_dataset(self, dataset_name, unique_ids, created_by):
        """Create a shared dataset from selected export results."""
        try:
            cursor = self.connection.cursor()
            
            # Create shared dataset entry
            cursor.execute("""
                INSERT INTO shared_datasets (dataset_name, created_by)
                VALUES (%s, %s)
            """, (dataset_name, created_by))
            
            dataset_id = cursor.lastrowid
            
            # Link export results to shared dataset
            for unique_id in unique_ids:
                cursor.execute("""
                    INSERT INTO shared_dataset_items (dataset_id, export_unique_id)
                    VALUES (%s, %s)
                """, (dataset_id, unique_id))
            
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error creating shared dataset: {e}")
            return False


# Utility function to initialize database
def initialize_database(host='localhost', database='baingan_db', user='root', password=''):
    """Initialize the database and create tables."""
    db = DatabaseManager(host, database, user, password)
    if db.connect():
        print("Connected to database successfully")
        if db.create_tables():
            print("Tables created successfully")
            db.disconnect()
            return True
        else:
            print("Failed to create tables")
            db.disconnect()
            return False
    else:
        print("Failed to connect to database")
        return False