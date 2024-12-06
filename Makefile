# Target to run the application
run:
	python3 app.py

# Install required Python packages
install:
	pip3 install -r requirements.txt

# Clean temporary or cached files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -f *.pyc *.pyo
