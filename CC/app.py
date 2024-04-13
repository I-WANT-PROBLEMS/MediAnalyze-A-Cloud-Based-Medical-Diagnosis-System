from flask import Flask, render_template, request, redirect, url_for, session
from flask_cognito import CognitoAuth
import requests
import boto3

app = Flask(__name__)
app.secret_key = 'team_16'  # Change to your secret key

# AWS Configuration
app.config['AWS_REGION'] = 'your_region'
app.config['DYNAMODB_TABLE'] = 'patient_records'

# Initialize a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name=app.config['AWS_REGION'])
table = dynamodb.Table(app.config['DYNAMODB_TABLE'])

# AWS Cognito Configuration
app.config['COGNITO_REGION'] = 'us-east-1'
app.config['COGNITO_USERPOOL_ID'] = 'us-east-1_nPUY9Z3ft'
app.config['COGNITO_APP_CLIENT_ID'] = '7tnfq9g45l68lvif91p9p7e1sp'
app.config['COGNITO_APP_CLIENT_SECRET'] = '18qqdgs73ao2i6k2um89tsuq6hbl0f55lr9aceeiap55p322ilnd'
app.config['COGNITO_DOMAIN'] = 'https://medical-app.auth.us-east-1.amazoncognito.com'
app.config['REDIRECT_URI'] = 'http://localhost:8000/aws_cognito_redirect'
app.config['LOGOUT_URI'] = 'http://localhost:8000/'

cogauth = CognitoAuth(app)


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/login')
def login():
    cognito_login_url = f"{app.config['COGNITO_DOMAIN']}/login?response_type=code&client_id={app.config['COGNITO_APP_CLIENT_ID']}&redirect_uri={app.config['REDIRECT_URI']}"
    return redirect(cognito_login_url)


@app.route('/aws_cognito_redirect')
def aws_cognito_redirect():
    code = request.args.get('code')
    if code:
        token_url = f"{app.config['COGNITO_DOMAIN']}/oauth2/token"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + app.config['COGNITO_APP_CLIENT_SECRET']
        }
        data = {
            'grant_type': 'authorization_code',
            'client_id': app.config['COGNITO_APP_CLIENT_ID'],
            'code': code,
            'redirect_uri': app.config['REDIRECT_URI']
        }
        response = requests.post(token_url, headers=headers, data=data)
        tokens = response.json()
        session['access_token'] = tokens.get('access_token')
        return redirect(url_for('main'))
    else:
        return "Authorization code not found", 400


@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'access_token' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        return redirect(url_for('upload_file'))
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    file = request.files['file']
    model_type = request.form['model_type']
    url = f'http://3.87.124.251:5000/predict_{model_type}'

    files = {'image': file.read()}
    response = requests.post(url, files=files)
    result = response.json()

    return render_template('result.html', result=result)


@app.route('/logout')
def logout():
    logout_url = f"{app.config['COGNITO_DOMAIN']}/logout?client_id={app.config['COGNITO_APP_CLIENT_ID']}&logout_uri={app.config['LOGOUT_URI']}"
    session.clear()
    return redirect(logout_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)