/**
 * Medicinal Leaf Classifier - API Integration
 * Overrides the dummy UI functions to connect to the actual Flask API.
 */

window.handleLogin = async function (e) {
    e.preventDefault();
    const btn = document.getElementById('btn-login');
    const txt = document.getElementById('btn-login-text');
    const ico = document.getElementById('icon-login');
    const spin = document.getElementById('spinner-login');

    const user = document.getElementById('login-username').value;
    const pass = document.getElementById('login-password').value;

    btn.disabled = true;
    txt.textContent = 'Authenticating…';
    ico.style.display = 'none';
    spin.style.display = 'block';

    try {
        const res = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });
        const data = await res.json();

        if (res.ok && data.status === "success") {
            // Success! Move to dashboard
            window.showView('view-dashboard');
            document.getElementById('login-username').value = '';
            document.getElementById('login-password').value = '';
        } else {
            alert("Login Failed: " + (data.message || "Invalid credentials"));
        }
    } catch (err) {
        alert("Connection error!");
    } finally {
        // resetLoginBtn Equivalent
        btn.disabled = false;
        txt.textContent = 'Continue';
        ico.style.display = '';
        spin.style.display = 'none';
    }
};

window.handleRegister = async function (e) {
    e.preventDefault();
    const btn = document.getElementById('btn-register');
    const txt = document.getElementById('btn-register-text');
    const ico = document.getElementById('icon-register');
    const spin = document.getElementById('spinner-register');

    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const pass = document.getElementById('register-password').value;

    btn.disabled = true;
    txt.textContent = 'Creating account…';
    ico.style.display = 'none';
    spin.style.display = 'block';

    try {
        const res = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password: pass })
        });
        const data = await res.json();

        if (res.ok && data.status === "success") {
            alert("Registration successful! Please log in.");
            window.switchTab('login');
            document.getElementById('form-register').reset();
        } else {
            alert("Registration Failed: " + (data.message || "User may already exist"));
        }
    } catch (err) {
        alert("Connection error!");
    } finally {
        // resetRegisterBtn Equivalent
        btn.disabled = false;
        txt.textContent = 'Create Account';
        ico.style.display = '';
        spin.style.display = 'none';
    }
};

window.logout = async function () {
    await fetch('/api/logout', { method: 'POST' });

    window.selectedFile = null;
    document.getElementById('drop-preview').style.display = 'none';
    document.getElementById('drop-default').style.display = 'block';
    window.enablePredict(false);
    window.showView('view-auth');
};

window.handlePredict = async function () {
    if (!window.selectedFile) return;

    const btn = document.getElementById('btn-predict');
    const txt = document.getElementById('btn-predict-text');
    const ico = document.getElementById('icon-predict');
    const spin = document.getElementById('spinner-predict');

    btn.disabled = true;
    txt.textContent = 'Analyzing…';
    ico.style.display = 'none';
    spin.style.display = 'block';

    const formData = new FormData();
    formData.append('imageFile', window.selectedFile);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (res.status === 401) {
            alert("Session expired, please login again.");
            window.logout();
            return;
        }

        const data = await res.json();

        if (res.ok && data.status === "success") {
            const p = data.prediction;

            // Update the DOM IDs!
            document.getElementById('plant-name').textContent = p.plant_name;
            document.getElementById('plant-botanical').textContent = p.botanical_name;
            document.getElementById('plant-uses').innerHTML = p.uses;
            document.getElementById('plant-side-effects').innerHTML = p.side_effects;
            document.getElementById('plant-remedies').innerHTML = p.remedies;

            // Confidence Logic
            const conf = p.confidence * 100; // API returns 0-1 range
            const pctStr = conf.toFixed(1) + '%';

            document.getElementById('conf-pct').textContent = pctStr;
            document.getElementById('result-confidence-badge').textContent = pctStr + ' confidence';

            // Handle warning threshold
            const warnBox = document.getElementById('warn-low-confidence');
            if (p.warning) {
                warnBox.style.display = 'flex';
                warnBox.querySelector('p').innerHTML = `<strong>Low confidence detected.</strong> ${p.warning}`;
            } else {
                warnBox.style.display = 'none';
            }

            window.showView('view-results');

            // Animate confidence bar after view transition
            setTimeout(() => {
                document.getElementById('conf-fill').style.width = pctStr;
            }, 500);

        } else {
            alert("Prediction Failed: " + (data.message || "Server Error"));
        }
    } catch (err) {
        alert("Connection error during prediction!");
        console.error(err);
    } finally {
        btn.disabled = false;
        txt.textContent = 'Predict & Analyze';
        ico.style.display = '';
        spin.style.display = 'none';
    }
};
