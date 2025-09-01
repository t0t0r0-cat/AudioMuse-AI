// /static/collection_script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let pb = null; // PocketBase instance
    let currentTaskId = null;
    let statusInterval = null;

    // --- DOM Element References ---
    // Login Section
    const loginSection = document.getElementById('login-section');
    const githubLoginBtn = document.getElementById('github-login-btn');
    const pocketbaseUrlInput = document.getElementById('pocketbase-url');
    const privacyCheckbox = document.getElementById('privacy-ack');
    const privacyLink = document.getElementById('privacy-link');

    // App Content
    const appContent = document.getElementById('app-content');
    const userInfoDisplay = document.getElementById('user-info');
    const logoutBtn = document.getElementById('logout-btn');
    const startSyncBtn = document.getElementById('start-sync-btn');
    const cancelSyncBtn = document.getElementById('cancel-sync-btn');

    // Task Status Display
    const statusTaskId = document.getElementById('status-task-id');
    const statusRunningTime = document.getElementById('status-running-time');
    const statusTaskType = document.getElementById('status-task-type');
    const statusStatus = document.getElementById('status-status');
    const statusProgress = document.getElementById('status-progress');
    const progressBar = document.getElementById('progress-bar');
    const statusLog = document.getElementById('status-log');
    const statusDetails = document.getElementById('status-details');

    // --- Core Functions ---

    /**
     * Initializes the PocketBase SDK and updates the UI based on auth state.
     */
    function initialize() {
        try {
            const url = pocketbaseUrlInput.value || localStorage.getItem('pocketbaseUrl');
            if (url) {
                pocketbaseUrlInput.value = url;
                pb = new PocketBase(url);
            }
        } catch (error) {
            console.error("Failed to initialize PocketBase:", error);
            showMessageBox("Error", "Invalid PocketBase URL.");
            pb = null;
        }
        updateUI();
    }

    /**
     * Toggles visibility of UI sections based on login status.
     */
    function updateUI() {
        if (pb && pb.authStore.isValid) {
            loginSection.style.display = 'none';
            appContent.style.display = 'block';
            const user = pb.authStore.model;
            userInfoDisplay.textContent = user.email || user.username || user.id;
            checkForExistingTask();
        } else {
            loginSection.style.display = 'block';
            appContent.style.display = 'none';
            stopPolling();
        }
    }

    // --- Auth Functions ---

    async function loginWithGithub() {
        const url = pocketbaseUrlInput.value;
        if (!url) {
            showMessageBox('Error', 'Please enter the PocketBase Server URL.');
            return;
        }
        
        // Re-initialize in case the URL was changed
        try {
             pb = new PocketBase(url);
        } catch(e) {
             showMessageBox('Error', 'Invalid PocketBase URL provided.');
             return;
        }

        try {
            await pb.collection('users').authWithOAuth2({ provider: 'github' });
            localStorage.setItem('pocketbaseUrl', url);
            showMessageBox('Success', 'Successfully logged in with GitHub.');
            updateUI();
        } catch (error) {
            console.error('GitHub login failed:', error);
            showMessageBox('Login Failed', 'Could not authenticate with GitHub. Check console for details.');
            pb.authStore.clear(); // Clear any partial auth state
            updateUI();
        }
    }

    function logout() {
        if (pb) {
            pb.authStore.clear();
        }
        showMessageBox('Logged Out', 'You have been successfully logged out.');
        updateUI();
    }


    // --- Task Management Functions (largely unchanged) ---

    function formatRunningTime(totalSeconds) {
        if (totalSeconds === null || totalSeconds === undefined || isNaN(totalSeconds) || totalSeconds < 0) {
            return '-- : -- : --';
        }
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const pad = (num) => String(num).padStart(2, '0');
        return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    }

    function displayTaskStatus(task) {
        if (!task || !task.task_id) {
            statusTaskId.textContent = 'N/A';
            statusRunningTime.textContent = '-- : -- : --';
            statusTaskType.textContent = 'N/A';
            statusStatus.textContent = 'IDLE';
            statusProgress.textContent = '0';
            progressBar.style.width = '0%';
            statusLog.textContent = 'No active or recent task found.';
            statusDetails.textContent = '';
            statusStatus.className = 'status-text status-idle';
            return;
        }

        statusTaskId.textContent = task.task_id;
        statusRunningTime.textContent = formatRunningTime(task.running_time_seconds);
        statusTaskType.textContent = task.task_type_from_db || task.task_type || 'N/A';
        const stateUpper = (task.state || task.status || 'IDLE').toUpperCase();
        statusStatus.textContent = stateUpper;
        statusProgress.textContent = task.progress || 0;
        progressBar.style.width = `${task.progress || 0}%`;

        let statusClass = 'status-idle';
        if (['SUCCESS', 'FINISHED'].includes(stateUpper)) statusClass = 'status-success';
        else if (['FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) statusClass = 'status-failure';
        else if (['PENDING', 'STARTED', 'PROGRESS'].includes(stateUpper)) statusClass = 'status-pending';
        statusStatus.className = `status-text ${statusClass}`;
        
        let statusMessage = 'N/A';
        if (task.details && typeof task.details === 'object') {
             statusMessage = (Array.isArray(task.details.log) && task.details.log.length > 0) 
                ? task.details.log[task.details.log.length - 1] 
                : (task.details.message || 'No message.');
        } else if (task.details) {
            statusMessage = task.details.toString();
        }
        statusLog.textContent = statusMessage;
        statusDetails.textContent = typeof task.details === 'object' ? JSON.stringify(task.details, null, 2) : task.details;
    }

    function startPolling(taskId) {
        if (statusInterval) clearInterval(statusInterval);
        currentTaskId = taskId;
        updateButtonStates(true);

        const poll = async () => {
            if (!currentTaskId) return;
            try {
                const response = await fetch(`/api/status/${currentTaskId}`);
                if (!response.ok) throw new Error(`Server responded with status ${response.status}`);
                const taskStatus = await response.json();
                displayTaskStatus(taskStatus);
                const stateUpper = (taskStatus.state || 'UNKNOWN').toUpperCase();
                if (['SUCCESS', 'FINISHED', 'FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) {
                    stopPolling();
                    showMessageBox('Task Finished', `Task ${currentTaskId} completed with status: ${stateUpper}`);
                }
            } catch (error) {
                console.error('Error polling task status:', error);
                displayTaskStatus({ task_id: currentTaskId, status: 'ERROR', details: `Polling error: ${error.message}` });
                stopPolling();
            }
        };
        poll();
        statusInterval = setInterval(poll, 3000);
    }

    function stopPolling() {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
        updateButtonStates(false);
    }

    function updateButtonStates(isTaskRunning) {
        startSyncBtn.disabled = isTaskRunning;
        cancelSyncBtn.disabled = !isTaskRunning || !currentTaskId;
    }

    async function startSync() {
        if (!pb || !pb.authStore.isValid) {
            showMessageBox('Error', 'You are not logged in. Please log in first.');
            updateUI();
            return;
        }

        const payload = {
            url: pocketbaseUrlInput.value,
            token: pb.authStore.token, // Use the auth token instead of email/password
            num_albums: parseInt(document.getElementById('num-albums').value, 10)
        };

        updateButtonStates(true);

        try {
            const response = await fetch('/api/collection/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok || !result.task_id) {
                throw new Error(result.message || 'Failed to start the sync task.');
            }
            startPolling(result.task_id);
            showMessageBox('Task Started', `Synchronization task enqueued with ID: ${result.task_id}`);
        } catch (error) {
            console.error('Error starting sync task:', error);
            showMessageBox('Error', `Could not start sync task: ${error.message}`);
            updateButtonStates(false);
        }
    }

    async function cancelSync() {
        if (!currentTaskId) return;
        cancelSyncBtn.disabled = true;
        try {
            const response = await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' });
            const result = await response.json();
            if (!response.ok) throw new Error(result.message || 'Failed to send cancellation request.');
            showMessageBox('Cancellation Sent', result.message);
        } catch (error) {
            console.error('Error cancelling task:', error);
            showMessageBox('Error', `Could not cancel task: ${error.message}`);
            updateButtonStates(!!statusInterval);
        }
    }
    
    async function checkForExistingTask() {
        try {
            const response = await fetch('/api/collection/last_task');
            if (!response.ok) throw new Error('Failed to fetch last task status.');
            const task = await response.json();
            if (task && task.task_id && task.status !== 'NO_PREVIOUS_TASK') {
                currentTaskId = task.task_id;
                displayTaskStatus(task);
                const isRunning = ['PENDING', 'STARTED', 'PROGRESS'].includes((task.status || '').toUpperCase());
                if (isRunning) {
                    startPolling(task.task_id);
                } else {
                    updateButtonStates(false);
                }
            } else {
                 updateButtonStates(false);
            }
        } catch (error) {
            console.error('Error checking for existing task:', error);
            showMessageBox('Error', 'Could not retrieve status of the last sync task.');
            updateButtonStates(false);
        }
    }

    function showMessageBox(title, message) {
        const boxId = 'custom-message-box';
        document.getElementById(boxId)?.remove();
        const messageBox = document.createElement('div');
        messageBox.id = boxId;
        messageBox.style.cssText = 'position: fixed; top: 20px; right: 20px; background-color: #fff; color: #1F2937; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 1000; border: 1px solid #E5E7EB; max-width: 400px; text-align: left;';
        messageBox.innerHTML = `<h3 style="font-weight: 600; margin-top:0; margin-bottom: 10px; color: #111827;">${title}</h3><p style="margin:0;">${message}</p><button style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5rem; color: #9CA3AF; cursor: pointer;" onclick="this.parentNode.remove()">&times;</button>`;
        document.body.appendChild(messageBox);
        setTimeout(() => messageBox.remove(), 5000);
    }

    // --- Event Listeners & Initialization ---
    githubLoginBtn.addEventListener('click', loginWithGithub);
    logoutBtn.addEventListener('click', logout);
    startSyncBtn.addEventListener('click', startSync);
    cancelSyncBtn.addEventListener('click', cancelSync);

    // Privacy policy logic
    privacyLink.addEventListener('click', (e) => {
        // Allow the link to open in a new tab
        // Use a small timeout to check the box after the browser processes the click
        setTimeout(() => {
            if (!privacyCheckbox.checked) {
                privacyCheckbox.checked = true;
                // Manually trigger the change event to enable the button
                privacyCheckbox.dispatchEvent(new Event('change'));
            }
        }, 100);
    });

    privacyCheckbox.addEventListener('change', () => {
        githubLoginBtn.disabled = !privacyCheckbox.checked;
    });
    
    initialize(); // Initial setup on page load
});

