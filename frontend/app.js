// JavaScript for handling form submission via AJAX
$(document).ready(function() {
    $('form').submit(function(event) {
        event.preventDefault(); // Prevent default form submission
        $.ajax({
            url: '/ask', // URL to submit the form data
            type: 'post', // HTTP method
            data: $(this).serialize(), // Form data to be submitted
            success: function(data) { // Callback function on successful response
                $('#response').text(data.response); // Update response area with server response
            },
            error: function(xhr, status, error) { // Error handling
                console.error("Error:", error);
            }
        });
    });
});
