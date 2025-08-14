package io

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"

	// Import SQLite driver for testing
	_ "github.com/mattn/go-sqlite3"
)

func TestSQLDatabase(t *testing.T) {
	t.Run("SQLite Connection", func(t *testing.T) {
		// Create temporary SQLite database
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "test.db")

		// Create database connection
		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("Failed to open SQLite database: %v", err)
		}
		defer db.Close()

		// Test connection
		err = db.Ping()
		if err != nil {
			t.Fatalf("Failed to ping database: %v", err)
		}

		// Create test table
		_, err = db.Exec(`CREATE TABLE test_table (
			id INTEGER PRIMARY KEY,
			name TEXT,
			value REAL,
			active BOOLEAN
		)`)
		if err != nil {
			t.Fatalf("Failed to create table: %v", err)
		}

		// Insert test data
		testData := []struct {
			id     int
			name   string
			value  float64
			active bool
		}{
			{1, "Alice", 100.5, true},
			{2, "Bob", 200.7, false},
			{3, "Carol", 300.9, true},
		}

		for _, row := range testData {
			_, err = db.Exec("INSERT INTO test_table (id, name, value, active) VALUES (?, ?, ?, ?)",
				row.id, row.name, row.value, row.active)
			if err != nil {
				t.Fatalf("Failed to insert test data: %v", err)
			}
		}

		// Test ReadSQL
		df, err := ReadSQL("SELECT * FROM test_table ORDER BY id", db)
		if err != nil {
			t.Fatalf("Failed to read from SQL: %v", err)
		}

		// Verify data
		if df.Len() != 3 {
			t.Errorf("Expected 3 rows, got %d", df.Len())
		}

		if len(df.Columns()) != 4 {
			t.Errorf("Expected 4 columns, got %d", len(df.Columns()))
		}

		// Check first row values
		id, _ := df.IAt(0, 0)
		name, _ := df.IAt(0, 1)
		value, _ := df.IAt(0, 2)
		active, _ := df.IAt(0, 3)

		if id != int64(1) {
			t.Errorf("Expected id=1, got %v", id)
		}
		if name != "Alice" {
			t.Errorf("Expected name=Alice, got %v", name)
		}
		if value != 100.5 {
			t.Errorf("Expected value=100.5, got %v", value)
		}
		if active != true {
			t.Errorf("Expected active=true, got %v", active)
		}
	})

	t.Run("Write DataFrame to SQL", func(t *testing.T) {
		// Create test DataFrame
		idSer, _ := series.FromSlice([]float64{4, 5, 6}, nil, "id")
		nameSer, _ := series.FromSlice([]interface{}{"Dave", "Eve", "Frank"}, nil, "name")
		valueSer, _ := series.FromSlice([]float64{400.1, 500.2, 600.3}, nil, "value")

		df, err := dataframe.FromSeries([]*series.Series{idSer, nameSer, valueSer})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		// Create temporary database
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "write_test.db")

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("Failed to open database: %v", err)
		}
		defer db.Close()

		// Create table for writing
		_, err = db.Exec(`CREATE TABLE write_test (
			id REAL,
			name TEXT,
			value REAL
		)`)
		if err != nil {
			t.Fatalf("Failed to create table: %v", err)
		}

		// Write DataFrame to SQL
		err = WriteSQL(df, "write_test", db, &SQLWriteOptions{
			IfExists: "append",
		})
		if err != nil {
			t.Fatalf("Failed to write DataFrame to SQL: %v", err)
		}

		// Read back and verify
		readDF, err := ReadSQL("SELECT * FROM write_test ORDER BY id", db)
		if err != nil {
			t.Fatalf("Failed to read back from SQL: %v", err)
		}

		if readDF.Len() != 3 {
			t.Errorf("Expected 3 rows, got %d", readDF.Len())
		}

		// Verify first row
		firstID, _ := readDF.IAt(0, 0)
		firstName, _ := readDF.IAt(0, 1)
		firstValue, _ := readDF.IAt(0, 2)

		if firstID != 4.0 {
			t.Errorf("Expected first ID=4.0, got %v", firstID)
		}
		if firstName != "Dave" {
			t.Errorf("Expected first name=Dave, got %v", firstName)
		}
		if firstValue != 400.1 {
			t.Errorf("Expected first value=400.1, got %v", firstValue)
		}
	})

	t.Run("SQL Write Options", func(t *testing.T) {
		// Test different write options (replace, append, fail)
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "options_test.db")

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("Failed to open database: %v", err)
		}
		defer db.Close()

		// Create test data
		ser1, _ := series.FromSlice([]float64{1, 2}, nil, "col1")
		df1, _ := dataframe.FromSeries([]*series.Series{ser1})

		ser2, _ := series.FromSlice([]float64{3, 4}, nil, "col1")
		df2, _ := dataframe.FromSeries([]*series.Series{ser2})

		// Test initial write
		err = WriteSQL(df1, "options_test", db, &SQLWriteOptions{
			IfExists:    "fail",
			CreateTable: true,
		})
		if err != nil {
			t.Fatalf("Failed initial write: %v", err)
		}

		// Test append
		err = WriteSQL(df2, "options_test", db, &SQLWriteOptions{
			IfExists: "append",
		})
		if err != nil {
			t.Fatalf("Failed to append: %v", err)
		}

		// Verify 4 rows total
		result, err := ReadSQL("SELECT COUNT(*) as count FROM options_test", db)
		if err != nil {
			t.Fatalf("Failed to count rows: %v", err)
		}
		count, err := result.IAt(0, 0)
		if err != nil {
			t.Fatalf("Failed to get count: %v", err)
		}
		// SQLite COUNT(*) may return string, so check flexibly
		var countVal int64
		switch v := count.(type) {
		case int64:
			countVal = v
		case string:
			if v == "4" {
				countVal = 4
			}
		case float64:
			countVal = int64(v)
		}
		if countVal != 4 {
			t.Errorf("Expected 4 rows after append, got %v", count)
		}

		// Test replace
		err = WriteSQL(df1, "options_test", db, &SQLWriteOptions{
			IfExists:    "replace",
			CreateTable: true,
		})
		if err != nil {
			t.Fatalf("Failed to replace: %v", err)
		}

		// Should have only 2 rows now
		result, err = ReadSQL("SELECT COUNT(*) as count FROM options_test", db)
		if err != nil {
			t.Fatalf("Failed to count rows after replace: %v", err)
		}
		count, err = result.IAt(0, 0)
		if err != nil {
			t.Fatalf("Failed to get count after replace: %v", err)
		}
		// SQLite COUNT(*) may return string, so check flexibly
		var countVal2 int64
		switch v := count.(type) {
		case int64:
			countVal2 = v
		case string:
			if v == "2" {
				countVal2 = 2
			}
		case float64:
			countVal2 = int64(v)
		}
		if countVal2 != 2 {
			t.Errorf("Expected 2 rows after replace, got %v", count)
		}
	})

	t.Run("Complex Query Types", func(t *testing.T) {
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "complex_test.db")

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("Failed to open database: %v", err)
		}
		defer db.Close()

		// Create tables with relationships
		_, err = db.Exec(`
			CREATE TABLE customers (
				id INTEGER PRIMARY KEY,
				name TEXT,
				email TEXT
			);
			CREATE TABLE orders (
				id INTEGER PRIMARY KEY,
				customer_id INTEGER,
				amount REAL,
				order_date TEXT,
				FOREIGN KEY(customer_id) REFERENCES customers(id)
			);
		`)
		if err != nil {
			t.Fatalf("Failed to create tables: %v", err)
		}

		// Insert sample data
		_, err = db.Exec(`
			INSERT INTO customers (id, name, email) VALUES 
			(1, 'John Doe', 'john@example.com'),
			(2, 'Jane Smith', 'jane@example.com');
			
			INSERT INTO orders (id, customer_id, amount, order_date) VALUES
			(1, 1, 100.50, '2024-01-01'),
			(2, 1, 200.75, '2024-01-02'),
			(3, 2, 150.25, '2024-01-03');
		`)
		if err != nil {
			t.Fatalf("Failed to insert sample data: %v", err)
		}

		// Test JOIN query
		joinQuery := `
			SELECT c.name as customer_name, c.email, o.amount, o.order_date
			FROM customers c
			JOIN orders o ON c.id = o.customer_id
			ORDER BY o.order_date
		`

		df, err := ReadSQL(joinQuery, db)
		if err != nil {
			t.Fatalf("Failed to execute JOIN query: %v", err)
		}

		if df.Len() != 3 {
			t.Errorf("Expected 3 rows from JOIN, got %d", df.Len())
		}

		if len(df.Columns()) != 4 {
			t.Errorf("Expected 4 columns from JOIN, got %d", len(df.Columns()))
		}

		// Test aggregate query
		aggQuery := `
			SELECT c.name, COUNT(o.id) as order_count, SUM(o.amount) as total_amount
			FROM customers c
			LEFT JOIN orders o ON c.id = o.customer_id
			GROUP BY c.id, c.name
			ORDER BY total_amount DESC
		`

		aggDF, err := ReadSQL(aggQuery, db)
		if err != nil {
			t.Fatalf("Failed to execute aggregate query: %v", err)
		}

		if aggDF.Len() != 2 {
			t.Errorf("Expected 2 rows from aggregate query, got %d", aggDF.Len())
		}

		// Verify John Doe has highest total
		firstName, _ := aggDF.IAt(0, 0)
		if firstName != "John Doe" {
			t.Errorf("Expected John Doe first in aggregate results, got %v", firstName)
		}
	})

	t.Run("Connection Pool Test", func(t *testing.T) {
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "pool_test.db")

		// Test connection pool settings
		pool, err := NewSQLConnectionPool("sqlite3", dbPath, &SQLPoolOptions{
			MaxOpenConns:    5,
			MaxIdleConns:    2,
			ConnMaxLifetime: 300, // 5 minutes
		})
		if err != nil {
			t.Fatalf("Failed to create connection pool: %v", err)
		}
		defer pool.Close()

		// Verify pool settings
		stats := pool.DB.Stats()
		if stats.MaxOpenConnections != 5 {
			t.Errorf("Expected MaxOpenConnections=5, got %d", stats.MaxOpenConnections)
		}

		// Test concurrent operations would go here
		// For now, just verify the connection works
		_, err = pool.DB.Exec("CREATE TABLE pool_test (id INTEGER)")
		if err != nil {
			t.Fatalf("Failed to execute query on pooled connection: %v", err)
		}
	})

	t.Run("Error Handling", func(t *testing.T) {
		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "error_test.db")

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("Failed to open database: %v", err)
		}
		defer db.Close()

		// Test reading from non-existent table
		_, err = ReadSQL("SELECT * FROM non_existent_table", db)
		if err == nil {
			t.Error("Expected error reading from non-existent table")
		}

		// Test invalid SQL
		_, err = ReadSQL("INVALID SQL QUERY", db)
		if err == nil {
			t.Error("Expected error for invalid SQL")
		}

		// Test writing to non-existent table with CreateTable=false
		ser, _ := series.FromSlice([]float64{1, 2}, nil, "col1")
		df, _ := dataframe.FromSeries([]*series.Series{ser})

		err = WriteSQL(df, "non_existent", db, &SQLWriteOptions{
			IfExists:    "fail",
			CreateTable: false,
		})
		if err == nil {
			t.Error("Expected error writing to non-existent table")
		}
	})
}

func TestSQLUtilities(t *testing.T) {
	t.Run("SQL Type Inference", func(t *testing.T) {
		// Test type inference for SQL column creation
		ser1, _ := series.FromSlice([]float64{1.5, 2.7}, nil, "float_col")
		ser2, _ := series.FromSlice([]interface{}{"text1", "text2"}, nil, "text_col")

		df, _ := dataframe.FromSeries([]*series.Series{ser1, ser2})

		sqlTypes := inferSQLTypes(df)

		if sqlTypes["float_col"] != "REAL" {
			t.Errorf("Expected REAL for float_col, got %s", sqlTypes["float_col"])
		}

		if sqlTypes["text_col"] != "TEXT" {
			t.Errorf("Expected TEXT for text_col, got %s", sqlTypes["text_col"])
		}
	})

	t.Run("Connection String Parsing", func(t *testing.T) {
		// Test various connection string formats
		testCases := []struct {
			driver     string
			connString string
			shouldWork bool
		}{
			{"sqlite3", ":memory:", true},
			{"sqlite3", "/tmp/test.db", true},
			{"postgres", "postgres://user:pass@localhost/db", true},
			{"mysql", "user:pass@tcp(localhost:3306)/db", true},
			{"invalid", "invalid://connection", false},
		}

		for _, tc := range testCases {
			valid := isValidConnectionString(tc.driver, tc.connString)
			if valid != tc.shouldWork {
				t.Errorf("Connection string validation failed for %s:%s, expected %v, got %v",
					tc.driver, tc.connString, tc.shouldWork, valid)
			}
		}
	})
}
