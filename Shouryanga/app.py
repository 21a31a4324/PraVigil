# Title : Intelligent License plate reader
# FY25-Q1-S1-01/23-Ver:1.0
# Author : Ravella Nandini.
# Releases: FY25-Q1-S1-01/20-Ver:2.0-CHG-We have done paper work regarding our project work. 
# Releases: FY25-Q1-S1-01/21-Ver:1.0-CHG-Preparation of dataset and gathering prerequisites. 
# Releases: FY25-Q1-S2-01/22-Ver:2.0-CHG-Enhancement of admin login and registration page.  
# Releases: FY25-Q1-S3-01/24-Ver:2.0-CHG-Fixed bugs in conversion of data from login and registration page to csv.files.  
# Releases: FY25-Q1-S4-01/27-Ver:2.0-CHG-Enhancement of introduction about contact, services provide html pages and css design.  
# Releases: FY25-Q1-S5-01/28-Ver:2.0-CHG-Enhancement of user login page along with ID verification  and OTP generation.  
# Releases: FY25-Q1-S6-01/30-Ver:2.0-CHG-Enhancement of admins home page.  
# Releases: FY25-Q1-S7-01/31-Ver:2.0-CHG-Enhancement of residence suspect list and black list.  
# Releases: FY25-Q1-S8-02/03-Ver:2.0-CHG-Enhancement of access check functionality.  
# Releases: FY25-Q1-S9-02/05-Ver:2.0-CHG-Enhancement of profile functionality .  
# Releases: FY25-Q1-S10-02/07-Ver:2.0-CHG-Gathered a structured data for model training.  
# Releases: FY25-Q1-S11-02/10-Ver:2.0-CHG-Training and testing the model on different images for improved accuracy.  
# Releases: FY25-Q1-S12-02/11-Ver:2.0-CHG-Enhancement of non resident page for visitor access.  
# Releases: FY25-Q1-S13-02/12-Ver:2.0-CHG-Enhancement of resident ID verification and OTP verification for Access into community.
# Releases: FY25-Q1-S13-02/14-Ver:2.0-CHG-Enhancement of backend code for designing a responsive dashboard.
# Releases: FY25-Q1-S13-02/17-Ver:2.0-CHG-Enhancement Displayed  statistics from csv files
# Releases: FY25-Q1-S13-02/18-Ver:2.0-CHG-Enhancement of model training for accurate results and it is going on.

from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import random
import os
import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
from werkzeug.utils import secure_filename
from datetime import timedelta
import re


app = Flask(__name__)
app.secret_key = 'a3f1d9b4e5c6a7f8d9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2'  #we can Change this for production security of our application
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
name_regex = re.compile("^[A-Za-z]+$")  # Here I am making sure that name should contain only alphabets
password_regex = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{11}$")

#These are the File paths
ADMIN_FILE = 'admin_users.csv'
RESIDENTS_FILE = 'residents.csv'
UPLOAD_FOLDER='static/uploads'
ALLOWED_EXTENSIONS={'png','jpg','jpeg','gif'}
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
# Loading the YOLO model as I trained my model using the custom data set which I made my self by collecting different license plate images I used easy ocr
model = YOLO("weights/best.pt")


RESIDENTS_CSV = "residents.csv"

# Ensuring whether the upload directory is present or not
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def append_to_suspect_list(vehicle_no):
    """Append a vehicle number to the suspect list and track its count."""
    try:
        df = pd.read_csv("suspect_list.csv", dtype=str)
        vehicle_counts = df["vehicle_no"].value_counts().to_dict()
        count = vehicle_counts.get(vehicle_no, 0) + 1
    except FileNotFoundError:
        count = 1

    # upon otp failure vehicle no will be appended to suspect list
    with open("suspect_list.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if count==1:
            writer.writerow(["vehicle_no"])
        writer.writerow([vehicle_no])

    # Moving to block list if count of otp failures reaches 3
    if count >= 3:
        move_to_block_list(vehicle_no)
#Appending vehicle to blocklist upon multiple otp failures
def move_to_block_list(vehicle_no):
    """Move a vehicle from the suspect list to the block list."""
    try:
        block_df = pd.read_csv("block_list.csv", dtype=str)
        if vehicle_no in block_df["vehicle_no"].astype(str).str.strip().str.upper().values:
            return  # Vehicle already exsists in block list so no need to enter
    except FileNotFoundError:
        pass  

    # Appending to block list
    with open("block_list.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([vehicle_no])
        
#checking the filetype of uploaded image as we allow only few file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# This is the main functionality to detect and extract license plate number from uploaded image 
def detect_license_plate(image_path):
    """Perform YOLO detection and OCR extraction on an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    results = model(image)
    cropped_plate_path = None
    license_plate_text = None

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_plate = image[y1:y2, x1:x2]

            # we are saving the detected plate here for further use
            cropped_plate_path = os.path.join(app.config["UPLOAD_FOLDER"], "cropped_plate.jpg")
            cv2.imwrite(cropped_plate_path, cropped_plate)

            # Performing  OCR to detect the vehicle no
            reader = easyocr.Reader(["en"])
            ocr_results = reader.readtext(cropped_plate)
            license_plate_text = "".join([res[1].upper() for res in ocr_results]).strip()
            session["extracted_vehicle_no"] = license_plate_text
            session["cropped_plate_path"] = cropped_plate_path

    return license_plate_text, cropped_plate_path
#checking whether the vehicle already exsists in residents list
def check_resident(license_plate_text):
    
    try:
        df = pd.read_csv("residents.csv", dtype=str,encoding="utf-8")  # I am ensuring that data is read as strings
        if "vehicle_no" in df.columns:  # Checking if the column exists
            license_plate_text = license_plate_text.strip().upper() 
            return(license_plate_text in df["vehicle_no"].astype(str).str.strip().str.upper().values)
        else:
           
            return False
    except FileNotFoundError:
        
        return False
    except Exception as e:
        
        return False
#checks whether the vehicle already in blocklist
def check_blocklist(license_plate_text):
   
    try:
        # Reading the CSV file, ensuring all data is treated as strings
        df = pd.read_csv("block_list.csv", dtype=str,encoding="utf-8")
        
        # Ensure the 'vehicle_no' column exists in my csv file
        if "vehicle_no" in df.columns:
            # Clean the license plate that is removing the gaps and converting to uppercase  text and checking if it exists in the blocklist
            license_plate_text = license_plate_text.strip().upper()
            return license_plate_text in df["vehicle_no"].astype(str).str.strip().str.upper().values
        else:
            
            return False
    except FileNotFoundError:
        
        return False
    except Exception as e:
        
        return False

#counting no of entries in each csv file to display in dashboard
def count_entries(csv_file):
   
    if not os.path.exists(csv_file):
        return 0  # if File does not exist, return 0

    try:
        # Reading only the necessary column to maintain transperacy
        df = pd.read_csv(csv_file, usecols=["vehicle_no"])
        
        # Counting the non-null entries in the 'vehicle_no' column
        count = df["vehicle_no"].notnull().sum()
        return count  # Returning the count of non-null vehicle numbers to display in dashboard
    except Exception as e:
      
        
        return 0  # Return 0 if there is an error

#Appending the access given vehicles to csv file
def append_to_csv(file_path, vehicle_no):
    
    # Checking if the file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # If the vehicle number already exists, return without adding into the csv file
        if vehicle_no in df["vehicle_no"].values:
            flash(f"Vehicle number {vehicle_no} already exists. Skipping entry.")
            return
    
    # Appending the new vehicle number to csv file
    new_data = pd.DataFrame([[vehicle_no]], columns=["vehicle_no"])
    new_data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    flash(f"Vehicle number {vehicle_no} added successfully.")
    
#accepts the file uploads from the user 
@app.route("/upload", methods=["POST"])
def upload_file():
    
    if "file" not in request.files:
        return render_template("access.html", message="No file part")

    file = request.files["file"]
    if file.filename == "":
        return render_template("access.html", message="No selected file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Perform detection and OCR on the uploaded file
        license_plate_text, _ = detect_license_plate(file_path)
        print(license_plate_text)

        if license_plate_text:

            if check_resident(license_plate_text):
                append_to_csv("access_granted.csv", [license_plate_text])
                flash(f"Access granted to vehicle {license_plate_text}.", "success")
                return redirect(url_for("matched"))
            elif check_blocklist(license_plate_text):
                # Check if the vehicle no already in blocked.csv and directly restrict the vehicle
                return redirect(url_for("not_matched"))  
            else:
                return redirect(url_for("non_resident"))

        return render_template("access.html", message="License plate not detected")

    return render_template("access.html", message="Invalid file type")

#Html page for file upload by the user
@app.route('/access')
def access():
    return render_template('access.html')

# Ensuring the admin CSV exists
if not os.path.exists(ADMIN_FILE):
    with open(ADMIN_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "email", "password"])  # CSV Header for Admins in the community

# Ensuring the residents CSV exists
if not os.path.exists(RESIDENTS_FILE):
    with open(RESIDENTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Vehicle_no", "Name", "Resident_id","Phone_no","Flat_no"])  # CSV Header for Residents in the community

# Helper functions to interact with CSV files to read and write the necessary information to the csv files
def read_csv(file):
    data = []
    with open(file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def write_admin(name, email, password):
    with open("admin_users.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, email, password])

# Introduction Page
@app.route('/')
def introduction():
    return render_template('introduction.html')

#about page
@app.route('/about')
def about():
    return render_template('about.html')

#contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Security Page
@app.route('/security')
def security():
    return render_template('security.html')

# Admin Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if not name_regex.match(name):
            flash("Name should only contain alphabets.", "danger")
            return redirect(url_for('register'))
        if not password_regex.match(password):
            flash("Password must be exactly 11 characters long, with a mix of alphabets, numbers, and special characters.", "danger")
            return redirect(url_for('register'))

        admins = read_csv(ADMIN_FILE)
        if any(admin['email'] == email for admin in admins):
            flash("Email already registered. Try logging in instead.", "danger")
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        write_admin(name, email, hashed_password)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

# Admin Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        admins = read_csv(ADMIN_FILE)
        admin = next((a for a in admins if a['email'] == email), None)
        if not admin:
            flash("Email not found. Please register first.", "danger")
            return redirect(url_for('register'))
        if not check_password_hash(admin['password'], password):
            flash("Incorrect password. Please try again.", "danger")
            return redirect(url_for('login'))
        session['admin_email'] = admin['email']
        session.permanent = True
        flash("Login successful!", "success")
        return redirect(url_for('home'))
    return render_template('login.html')

# User Login
@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    otp_generated = False
    otp = None
    if request.method == 'POST':
        user_id = request.form['id']
        residents = read_csv(RESIDENTS_FILE)
        resident = next((r for r in residents if r['resident_id'] == user_id), None)
        if not resident:
            flash("ID not found in the residents list.", "danger")
            return redirect(url_for('user_login'))
        otp = random.randint(100000, 999999)
        session['otp'] = otp 
        session['user_id'] = user_id  
        otp_generated = True
        flash(f"OTP for login: {otp}", "success")
    return render_template('user_login.html', otp_generated=otp_generated, otp=otp)

# Home Page
@app.route('/home')
def home():
    if 'admin_email' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))
    return render_template('home.html')
#profile page
@app.route('/profile')
def profile():
    admin_email=session.get("admin_email")
    return render_template('profile.html',admin_email=admin_email)

#suspect display
@app.route('/suspect_list')
def suspect_list():
    data = read_csv('suspect_list.csv')
    return render_template('suspect_list.html', suspect_list=data)

#dashboard
@app.route('/dashboard')
def dashboard():
    
    # Count entries in each CSV file for creating a dashboard
    suspect_count = count_entries("suspect_list.csv")
    blocklist_count = count_entries("block_list.csv")
    access_granted_count = count_entries("access_granted.csv")
    residents_count=count_entries("residents.csv")
    
    # Pass counts to the template for display
    return render_template(
        'dashboard.html',
        suspect_count=suspect_count,
        blocklist_count=blocklist_count,
        access_granted_count=access_granted_count,
        residents_count=residents_count
    )

#Accessgranted
@app.route('/matched')
def matched():
    return render_template('matched.html')

#Acessdenied
@app.route('/not_matched')
def not_matched():
    return render_template('not_matched.html')
#reads file
def readd(file):
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Route to display the Non-Resident page
@app.route('/non_resident', methods=['GET', 'POST'])
def non_resident():
    # Reset session data for a new verification process of non residents
    if request.method == 'GET':
        session.pop('verification_stage', None)  # Clearing verification stage
        session.pop('resident_name', None)  # Clearing resident name
        session.pop('resident_id', None)  # Clearing resident ID
        session.pop('otp', None)  # Clearing OTP

    verification_stage = session.get('verification_stage', 'id')  # Default to ID verification

    if request.method == 'POST':
        if verification_stage == "id":  # ID Verification Stage
            user_input = request.form.get('id') 
            residents = read_csv(RESIDENTS_FILE)
            resident = next((r for r in residents if r['resident_id'] == user_input or r['phone_no'] == user_input), None)



            if resident:
                session['resident_name'] = resident['name']  # Storing resident name in session
                session['resident_id'] = user_input #storing resident ID in session
                verification_stage = "otp"  # Moving to OTP verification stage
                session['verification_stage'] = verification_stage  # Update the session
                flash("Please enter the OTP sent to you.", "success")
            else:
                flash("Invalid Community ID or Phone Number. Please try again.", "danger")

        elif verification_stage == "otp":  # OTP Verification Stage
            entered_otp = request.form.get('otp')
            stored_otp = session.get('otp')  # Retrieving OTP from session

            if entered_otp == str(stored_otp): # Comparing entered OTP with stored OTP
                vehicle_no = session.get('extracted_vehicle_no') 
                append_to_csv("access_granted.csv", [vehicle_no])
                flash("OTP verified successfully!", "success")
                return redirect(url_for('matched'))  # Redirect to matched page in case of sucessful verification
            else:
                vehicle_no = session.get('extracted_vehicle_no')  # Retrieving the vehicle number to add to suspect list
                if vehicle_no:
                    append_to_suspect_list(vehicle_no)
                    flash("Incorrect OTP. Vehicle added to suspect list.Try again!", "danger")
                
                return redirect(url_for('not_matched'))  # Redirect to not_matched page incase of otp failure

    return render_template('non_resident.html', verification_stage=verification_stage)

#blocklist display
@app.route('/block_list')
def block_list():
    data = read_csv('block_list.csv')
    return render_template('block_list.html', blocklist=data)

#residents display
@app.route('/residents')
def residents():
    data = read_csv('residents.csv')  # Read the CSV file
    return render_template('residents.html', residents=data)
#checking whether the resident exsists!
def resident_exists(resident_id, vehicle_no):
    """Check if the resident ID or vehicle number already exists in the CSV file."""
    with open(RESIDENTS_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        for row in reader:
            if row and (row[2] == resident_id or row[0] == vehicle_no):  # Compare ID or vehicle number
                return True
    return False

@app.route('/add_resident')
def add_resident_form():
    
    return render_template('add_resident.html')

@app.route('/add_resident', methods=['POST'])
def add_resident():
   
    data = request.json
    vehicle_no = data.get("vehicle_no", "").strip()
    name = data.get("name", "").strip()
    resident_id = data.get("resident_id", "").strip()
    phone_no = data.get("phone_no", "").strip()
    flat_no = data.get("flat_no", "").strip()

    # Validating the required fields
    if not all([vehicle_no, name, resident_id, phone_no, flat_no]):
        return jsonify({"message": "All fields are required!"}), 400

    if resident_exists(resident_id, vehicle_no):
        return jsonify({"message": "Resident with this ID or Vehicle No already exists!"}), 409

    
    with open(RESIDENTS_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([vehicle_no, name, resident_id, phone_no, flat_no])

    return jsonify({"message": "Resident added successfully!"}), 201

#deletion of a resident

def delete_entry(file_path, vehicle_no):
    
    rows = []
    deleted = False
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader if row and row[0] != vehicle_no]
        deleted = len(rows) < sum(1 for row in open(file_path, 'r'))
    
    if deleted:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        return True
    return False

@app.route('/remove_resident', methods=['POST'])
def handle_delete_resident():
    data = request.json
    vehicle_no = data.get("vehicle_no")
    list_type = data.get("list_type")

    file_mapping = {
        "resident": "residents.csv",
        "suspect": "suspect_list.csv",
        "blocklist": "block_list.csv"
    }

    if list_type in file_mapping and delete_entry(file_mapping[list_type], vehicle_no):
        return jsonify({"message": f"Vehicle {vehicle_no} deleted from {list_type} list."})
    return jsonify({"message": "Vehicle not found!"})

@app.route('/remove_resident')
def delete_resident_page():
    return render_template('remove_resident.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
