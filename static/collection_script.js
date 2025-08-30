// /static/collection_script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
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

    // --- State Variables ---
    let currentTaskId = null;
    let statusInterval = null;

    // --- Functions ---

    /**
     * Formats a duration in seconds into a HH:MM:SS string.
     * @param {number} totalSeconds The total seconds to format.
     * @returns {string} The formatted time string.
     */
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

    /**
     * Displays the status of a task in the UI.
     * @param {object} task - The task object with status details.
     */
    function displayTaskStatus(task) {
        if (!task || !task.task_id) {
            // Reset UI to default state if no task data
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
        // FIX: Check for 'state' from the polling API and 'status' from the initial load API.
        const stateUpper = (task.state || task.status || 'IDLE').toUpperCase();
        statusStatus.textContent = stateUpper;
        statusProgress.textContent = task.progress || 0;
        progressBar.style.width = `${task.progress || 0}%`;

        // Update status color
        let statusClass = 'status-idle';
        if (['SUCCESS', 'FINISHED'].includes(stateUpper)) {
            statusClass = 'status-success';
        } else if (['FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) {
            statusClass = 'status-failure';
        } else if (['PENDING', 'STARTED', 'PROGRESS'].includes(stateUpper)) {
            statusClass = 'status-pending';
        }
        statusStatus.className = `status-text ${statusClass}`;

        // Update log and details
        let statusMessage = 'N/A';
        if (task.details && typeof task.details === 'object') {
            statusMessage = (Array.isArray(task.details.log) && task.details.log.length > 0) ?
                task.details.log[task.details.log.length - 1] :
                (task.details.message || 'No message.');
        } else if (task.details) {
            statusMessage = task.details.toString();
        }
        statusLog.textContent = statusMessage;
        statusDetails.textContent = typeof task.details === 'object' ? JSON.stringify(task.details, null, 2) : task.details;
    }


    /**
     * Starts polling for the status of a given task ID.
     * @param {string} taskId - The ID of the task to poll.
     */
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
        
        poll(); // Immediate poll
        statusInterval = setInterval(poll, 3000);
    }

    /**
     * Stops the status polling interval.
     */
    function stopPolling() {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
        // Don't clear currentTaskId here, so we can still cancel a finished but failed task
        updateButtonStates(false);
    }

    /**
     * Updates the enabled/disabled state of the control buttons.
     * @param {boolean} isTaskRunning - Whether a task is currently active.
     */
    function updateButtonStates(isTaskRunning) {
        startSyncBtn.disabled = isTaskRunning;
        cancelSyncBtn.disabled = !isTaskRunning || !currentTaskId;
    }

    /**
     * Starts the synchronization task.
     */
    async function startSync() {
        const payload = {
            url: document.getElementById('pocketbase-url').value,
            email: document.getElementById('pocketbase-email').value,
            password: document.getElementById('pocketbase-password').value,
            num_albums: parseInt(document.getElementById('num-albums').value, 10)
        };

        if (!payload.url || !payload.email || !payload.password) {
            showMessageBox('Error', 'Please fill in all Pocketbase server details.');
            return;
        }

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

    /**
     * Cancels the currently running task.
     */
    async function cancelSync() {
        if (!currentTaskId) return;
        cancelSyncBtn.disabled = true;

        try {
            const response = await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.message || 'Failed to send cancellation request.');
            }
            showMessageBox('Cancellation Sent', result.message);
            // Polling will handle the final status update and stop itself.
        } catch (error) {
            console.error('Error cancelling task:', error);
            showMessageBox('Error', `Could not cancel task: ${error.message}`);
            // Re-enable button only if the task is still considered active.
            updateButtonStates(!!statusInterval);
        }
    }
    
    /**
     * Checks for the last known collection sync task on page load.
     */
    async function checkForExistingTask() {
        try {
            const response = await fetch('/api/collection/last_task');
            if (!response.ok) throw new Error('Failed to fetch last task status.');

            const task = await response.json();
            if (task && task.task_id && task.status !== 'NO_PREVIOUS_TASK') {
                currentTaskId = task.task_id;
                displayTaskStatus(task);

                const stateUpper = (task.status || '').toUpperCase();
                const isRunning = ['PENDING', 'STARTED', 'PROGRESS'].includes(stateUpper);
                if (isRunning) {
                    startPolling(task.task_id);
                } else {
                    updateButtonStates(false); // Task is finished, so buttons are in idle state
                }
            } else {
                 updateButtonStates(false); // No task found
            }
        } catch (error) {
            console.error('Error checking for existing task:', error);
            showMessageBox('Error', 'Could not retrieve status of the last sync task.');
            updateButtonStates(false);
        }
    }


    /**
     * Shows a temporary message box in the corner of the screen.
     * @param {string} title - The title of the message.
     * @param {string} message - The body of the message.
     */
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
    startSyncBtn.addEventListener('click', startSync);
    cancelSyncBtn.addEventListener('click', cancelSync);
    
    checkForExistingTask(); // Check for a task when the page loads
});

